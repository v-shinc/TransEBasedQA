#define ARMA_DONT_USE_CXX11
#include <armadillo>
#include <string>
#include <fstream>
#include <unordered_map>
#include <boost/algorithm/string.hpp>
#include <set>
#include <thread>
#include <random>
#include <numeric>
#include <cmath>

using namespace arma;
using namespace std;

class Blob{
	enum FIELD{ QUESTION = 3, SUBGRAPH = 4, TOPIC = 0, RELATION = 1, ANSWER = 2, OTHERANSWERS = 5 };
public:
	explicit Blob(const string& data);
	Blob(const Blob&) = delete;
	Blob(Blob && rhs) :
		question_indices(std::move(rhs.question_indices)),
		answer_indices(std::move(rhs.answer_indices)),
		other_answers_indices(std::move(rhs.other_answers_indices)),
		topic_entity(std::move(rhs.topic_entity)),
		answer_(std::move(rhs.answer_)),
		relation_pair(std::move(rhs.relation_pair))
	{}
	Blob &operator =(const Blob&) = delete;
	const vector<unsigned>& questionIndices(){ return question_indices; }
	const vector<unsigned>& answerIndices(){ return answer_indices; }
	const vector<unsigned> &otherAnswers(){ return other_answers_indices; }
	unsigned topicEntity(){ return topic_entity; }
	unsigned answer(){ return answer_; }
	pair<unsigned, unsigned> relation(){ return relation_pair; }

private:
	vector<unsigned> question_indices;
	vector<unsigned> answer_indices;
	vector<unsigned> other_answers_indices;
	unsigned topic_entity, answer_;
	pair<unsigned, unsigned> relation_pair;
};
Blob::Blob(const string& data){
	vector <string> fields;
	vector <string> tmp;
	boost::split(fields, data, boost::is_any_of("\t"));
	boost::split(tmp, fields[QUESTION], boost::is_any_of(" "));
	for (auto &s : tmp) question_indices.push_back(stoi(s));
	boost::split(tmp, fields[SUBGRAPH], boost::is_any_of(" "));
	for (auto &s : tmp) answer_indices.push_back(stoi(s));
	topic_entity = stoi(fields[TOPIC]);
	answer_ = stoi(fields[ANSWER]);
	auto found = fields[RELATION].find(" ");
	if (found != string::npos){
		relation_pair = make_pair(stoi(fields[RELATION].substr(0, found)), stoi(fields[RELATION].substr(found + 1)));
	}
	else{
		relation_pair = make_pair(stoi(fields[RELATION]), -1);
	}

	//if (relationStr == "") cout << "hhh" <<data << endl;
	if (fields[OTHERANSWERS] != "None"){
		boost::split(tmp, fields[OTHERANSWERS], boost::is_any_of(" "));
		for (auto &s : tmp) other_answers_indices.push_back(stoi(s));
	}
}
class QAEmbedding{
public:
	enum class AnswerMode{ SINGLE, PATH, SUBGRAPH };

	QAEmbedding(int dimension, const string &trainDataPath,
		const string &resultDirectory, const string& wordListPath, const string&kbListPath, const string&fbPath, AnswerMode _mode);
	void train(int n_epoch, double alpha, double gamma);
	void save(){
		Ww.save(out_dir + "/wordEmbedding_s.out");
		Ws.save(out_dir + "/kbEmbedding_s.out");
	}
private:
	AnswerMode mode;
	int dim, Nw, Ns;
	string train_path, out_dir;
	mat Ww, Ws;
	double alpha, gamma;

	//unordered_map<string, unsigned> index_of_word;
	unordered_map<string, unsigned> index_of_kb;
	unordered_map <unsigned, unordered_map<unsigned, vector<unsigned>>> graph;
	void train_one(Blob &blob, double & loss);
	void subgraphPresentation(unsigned root, vector<unsigned> &indices);
	double gradientDescent(const vector <unsigned>&ques_indices, const vector<unsigned> &answer_indices, const vector<unsigned> &wrong_answer_indices);
	static uniform_real_distribution<double> distribution;
	static default_random_engine generator;
};
uniform_real_distribution<double> QAEmbedding::distribution(0.0, 1.0);
default_random_engine QAEmbedding::generator;
QAEmbedding::QAEmbedding(int dimension, const string &trainDataPath,
	const string &resultDirectory, const string& wordListPath, const string&kbListPath, const string&fbPath, AnswerMode _mode = AnswerMode::SUBGRAPH)
	:dim(dimension), train_path(trainDataPath), out_dir(resultDirectory), mode(_mode){
	string sline;
	unsigned no = 0;
	fstream word_fin(wordListPath);
	while (getline(word_fin, sline)){
		//index_of_word[sline] = no++;
		++no;
	}
	word_fin.close();
	//cout << "size of index_of_word " << no << " " << index_of_word.size() << endl;

	unsigned kb_no = 0;
	fstream kb_fin(kbListPath);
	while (getline(kb_fin, sline)){
		index_of_kb[sline] = kb_no++;
	}
	kb_fin.close();
	cout << "size of index_of_kb " << kb_no << " " << index_of_kb.size() << endl;

	fstream fb_fin(fbPath);
	while (getline(fb_fin, sline)){
		vector<string> strs;
		boost::split(strs, sline, boost::is_any_of("\t"));
		graph[index_of_kb[strs[0]]][index_of_kb[strs[1]]].push_back(index_of_kb[strs[2]]);
	}
	fb_fin.close();
	cout << "Finish loading graph\n";
	Nw = no; // index_of_word.size();
	Ns = index_of_kb.size();
	Ww = normalise(randu<mat>(dim, Nw));
	Ws = normalise(randu<mat>(dim, Ns * 2));
	
}
//ques_indices: indicate the index of each word appearing in the question in sequence
double QAEmbedding::gradientDescent(const vector <unsigned>&ques_indices, const vector<unsigned> &answer_indices, const vector<unsigned> &wrong_answer_indices){

	vec wq = zeros<vec>(dim);
	vec wp = zeros<vec>(dim);
	vec wn = zeros<vec>(dim);

	for (auto i : ques_indices){ 

		if (norm(Ww.col(i)) > 1) Ww.col(i) = normalise(Ww.col(i));
		wq += Ww.col(i);
	} 
	
	for (auto i : answer_indices){

		if (norm(Ws.col(i)) > 1) Ws.col(i) = normalise(Ws.col(i));
		wp += Ws.col(i);
	}
	
	for (auto i : wrong_answer_indices) {

		if (norm(Ws.col(i)) > 1) Ws.col(i) = normalise(Ws.col(i));
		wn += Ws.col(i);
	}
	
	double loss = gamma - dot(wq, wp) + dot(wq, wn);
	if (loss < 0) return 0;
	for (auto i : answer_indices){

		Ws.col(i) += alpha * wq;
	}
	for (auto i : ques_indices){

		Ww.col(i) += alpha * wp;
	}
	for (auto i : wrong_answer_indices){

		Ws.col(i) -= alpha * wq;
	}
	for (auto i : ques_indices){

		Ww.col(i) -= alpha * wn;
	}
	return loss;
}
void QAEmbedding::subgraphPresentation(unsigned root, vector<unsigned> &indices){

	auto start_index = index_of_kb.size();
	double prob = min(1., 100.0 / graph[root].size());
	set<unsigned> subgraph_set;
	if (graph.count(root) == 0) { cout << "can't find " << root << " in graph" << endl; return; }
	for (auto & item : graph[root]){
		if (distribution(generator) <= prob){

			subgraph_set.insert(item.first + start_index);
			for (auto & o : item.second){

				if (o >= Ws.n_cols)cout << o << endl;

				subgraph_set.insert(o + start_index);
			}
		}
	}
	copy(subgraph_set.begin(), subgraph_set.end(), back_inserter(indices));
}
void QAEmbedding::train_one(Blob &blob, double &loss){
	auto q_entity = blob.topicEntity();
	int n_hop = 1;
	
	auto correct_relation = blob.relation();
	auto start_index = index_of_kb.size();
	
	static uniform_real_distribution<double> distribution(0.0, 1.0);
	static default_random_engine generator;

	for (auto &item : graph[q_entity]){
		
		// 1-hop negative data
		if (!(item.first == correct_relation.first && correct_relation.second == -1)) {
			
			for (auto &wa : item.second){
				// sampling 50 % of time from the set of entities connected to the entity of the question 
				if (distribution(generator) <= 0.5){

					vector <unsigned> wa_indices;
					wa_indices.push_back(q_entity);
					wa_indices.push_back(item.first);
					wa_indices.push_back(wa);
					subgraphPresentation(wa, wa_indices);
					loss += gradientDescent(blob.questionIndices(), blob.answerIndices(), wa_indices);
				}
			}
		}
		//if (distribution(generator) > 0.5) continue;
		//// 2-hop negative data
		//auto mediator_index = rand() % item.second.size();
		//for (auto &mediate : item.second){
		//	if (graph.count(mediate) == 0) continue;
		//	for (auto &item2 : graph[mediate]){
		//		if (item.first + " " + item2.first == correct_relation) continue;
		//		neighbor_count_2_hop += item2.second.size();
		//		for (auto &wa : item2.second){
		//			
		//			if (distribution(generator) <= 0.3){
		//				vector <unsigned> wa_indices;
		//				// path representation
		//				wa_indices.push_back(index_of_kb[q_entity]);
		//				wa_indices.push_back(index_of_kb[item.first]);
		//				wa_indices.push_back(index_of_kb[item2.first]);
		//				wa_indices.push_back(index_of_kb[wa]);
		//				
		//				// add subgraph
		//				if (graph.count(wa) > 0){
		//					// use at most 100 paths in the subgraph representation
		//					auto total_path_num = graph[wa].size();
		//					double prob = min(1., 100.0 / total_path_num);
		//					set<unsigned> subgraph_set;
		//					for (auto & item3 : graph[wa]){
		//						if (distribution(generator) <= prob){
		//							subgraph_set.insert(index_of_kb[item3.first] + start_index);
		//							for (auto & o : item3.second){
		//								subgraph_set.insert(index_of_kb[o] + start_index);
		//							}
		//						}
		//					}
		//					copy(subgraph_set.begin(), subgraph_set.end(), back_inserter(wa_indices));
		//				}
		//				
		//				loss += gradientDescent(blob.questionIndices(), blob.answerIndices(), wa_indices);
		//			}
		//		}
		//	}
		//}
	}
}
void QAEmbedding::train(int n_epoch, double _alpha, double _gamma){
	alpha = _alpha;
	gamma = _gamma;
	unsigned long n_workers = 30;
	
	vector<thread> threads(n_workers - 1);
	string sline;
	double total_loss;
	/*vector< unordered_map<unsigned, vec>> grad_kb_s(n_workers);
	vector< unordered_map<unsigned, vec>> grad_word_s(n_workers);*/
	vector<Blob> blob_vec;
	fstream fin(train_path);

	// read all training data
	while (getline(fin, sline)){
		blob_vec.push_back(std::move(Blob(sline)));
	}
	cout << "start to train\n";
	unsigned long block_size = blob_vec.size() / n_workers;
	
	for (int epoch_index = 1; epoch_index <= n_epoch; ++epoch_index){
		clock_t time_begin = clock();
		vector <double> loss_s(n_workers, 0);
		auto block_start = blob_vec.begin();

		for (unsigned long i = 0; i < n_workers - 1; ++i){
			auto block_end = block_start;
			std::advance(block_end, block_size);

			threads[i] = thread([&, block_start, block_end, i](){
				size_t n_block = block_end - block_start;
				for (int i = 0; i < n_block; ++i){
					auto it = block_start + size_t(n_block * distribution(generator));
					train_one(*it, loss_s[i]);
				}
				/*for (auto it = block_start; it != block_end; ++it)*/
					
			});

			block_start = block_end;
		}
		for (auto it = block_start; it != blob_vec.end(); ++it){
			train_one(*it, loss_s[n_workers - 1]);
		}
		for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
		total_loss = std::accumulate(loss_s.begin(), loss_s.end(), 0);
		cout << epoch_index << " epoch, loss = " << total_loss << ", it takes " << double(clock() - time_begin) / CLOCKS_PER_SEC << " s.\n";
		//printf("# %d epoch, loss = %f, it takes %f s\n", epoch_index, total_loss, double(clock() - time_begin) / CLOCKS_PER_SEC);
		save();
		fin.close();
	
	}

}
int main(int argc, char **argv){
	auto qae = QAEmbedding(64, "data/qa.train", "data/result.matrix", "data/word.list", "data/fb.entry.list", "data/fb.triple");
	qae.train(100, 0.001, 0.1);
	qae.save();
}
