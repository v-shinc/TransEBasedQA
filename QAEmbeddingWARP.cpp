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
		answerStr(std::move(rhs.answerStr)),
		relationStr(std::move(rhs.relationStr))
		{}
	Blob &operator =(const Blob&) = delete;
	const vector<unsigned>& questionIndices(){ return question_indices; }
	const vector<unsigned>& answerIndices(){ return answer_indices; }
	const vector<unsigned> &otherAnswers(){ return other_answers_indices; }
	const string &topicEntity(){ return topic_entity; }
	const string &answer(){ return answerStr; }
	const string &relation(){ return relationStr; }
	
private:
	vector<unsigned> question_indices;
	vector<unsigned> answer_indices;
	vector<unsigned> other_answers_indices;
	string topic_entity, answerStr, relationStr;
};
Blob::Blob(const string& data){
	vector <string> fields;
	vector <string> tmp;
	boost::split(fields, data, boost::is_any_of("\t"));
	boost::split(tmp, fields[QUESTION], boost::is_any_of(" "));
	for (auto &s : tmp) question_indices.push_back(stoi(s));
	boost::split(tmp, fields[SUBGRAPH], boost::is_any_of(" "));
	for (auto &s : tmp) answer_indices.push_back(stoi(s));
	topic_entity = fields[TOPIC];
	answerStr = fields[ANSWER];
	relationStr = fields[RELATION];
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
	unordered_map <string, unordered_map<string, vector<string>>> graph;
	unordered_map<string, vector<string>> entity_to_relation_vec;
	unordered_map <string, int> n_direct_neighbor;
	unordered_map <string, int> n_neighbor_within_2;
	unordered_map <string, unsigned> n_node_per_rel;
	void train_one(Blob &blob, double & loss);
	double gradientDescent(const vector <unsigned>&ques_indices, const vector<unsigned> &answer_indices, const vector<unsigned> &wrong_answer_indices, double weight);
	void randomNegativeData(const string &topic, const string& correct_rel, vector<unsigned> &indices);
	void subgraphPresentation(string &o, vector<unsigned> &indices);
	static const unsigned MAX_TABLE_SIZE = 2000;
	double l_table[MAX_TABLE_SIZE];
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
		graph[strs[0]][strs[1]].push_back(strs[2]);
	}
	for (auto &s_ros : graph){
		auto & s = s_ros.first;
		for (auto &r_os : s_ros.second){
			entity_to_relation_vec[s].push_back(r_os.first);
		}
	}
	fb_fin.close();
	Nw = no;// index_of_word.size();
	Ns = index_of_kb.size();
	Ww = normalise(randu<mat>(dim, Nw));
	Ws = normalise(randu<mat>(dim, Ns * 2));

	l_table[0] = 0;
	for (int i = 1; i < 1000; ++i){
		l_table[i] = l_table[i - 1] + 1. / i;
	}

	
	for (auto &item : graph){
		auto & root = item.first;
		for (auto &r_os : item.second){
			n_direct_neighbor[root] += r_os.second.size(); // direct neighbor
		}
	}
	for (auto &item : graph){
		auto & root = item.first;
		n_neighbor_within_2[root] += n_direct_neighbor[root];  // add number of direct neighbor
		for (auto &r_os : item.second){
			n_node_per_rel[root + r_os.first] += r_os.second.size();
			for (auto &o : r_os.second){
				n_node_per_rel[root + r_os.first] += n_direct_neighbor[o];
				n_neighbor_within_2[root] += n_direct_neighbor[o]; // add number of indirect neighbor
			}
		}
	}
	
}
//boost::minstd_rand QAEmbedding::rng(time(0));


//ques_indices: indicate the index of each word appearing in the question in sequence
//double QAEmbedding::gradientDescent(const vector <unsigned>&ques_indices, const vector<unsigned> &answer_indices, const vector<unsigned> &wrong_answer_indices){
//
//	vec wq = zeros<vec>(dim);
//	vec wp = zeros<vec>(dim);
//	vec wn = zeros<vec>(dim);
//	for (auto i : ques_indices){ 
//		//if (i >= Ww.n_cols)cout << i << "xxxx" << Ww.n_cols << endl;
//		vec t = Ww.col(i);
//		if (norm(t) > 1) t = normalise(t);
//		wq += t;
//	} 
//	
//	for (auto i : answer_indices){
//		//if (i >= Ws.n_cols)cout << i << "wwww" << Ws.n_cols << endl;
//		vec t = Ws.col(i);
//		if (norm(t) > 1) t = normalise(t);
//		wp += t; 
//	}
//	
//	for (auto i : wrong_answer_indices) {
//		//if (i >= Ws.n_cols)cout << i << "aaaa" << Ws.n_cols << endl;
//		vec t = Ws.col(i);
//		if (norm(t) > 1) t = normalise(t);
//		wn += t; 
//	}
//	
//	double loss = gamma - dot(wq, wp) + dot(wq, wn);
//	if (loss < 0) return 0;
//	for (auto i : answer_indices){
//		Ws.col(i) += alpha * wq;
//	}
//	for (auto i : ques_indices){
//		Ww.col(i) += alpha * wp;
//	}
//	for (auto i : wrong_answer_indices){
//		Ws.col(i) -= alpha * wq;
//	}
//	for (auto i : ques_indices){
//		Ww.col(i) -= alpha * wn;
//	}
//	return loss;
//}
double QAEmbedding::gradientDescent(const vector <unsigned>&ques_indices, const vector<unsigned> &answer_indices, const vector<unsigned> &wrong_answer_indices, double weight){
		vec f = zeros<vec>(dim);
		vec g1 = zeros<vec>(dim);
		vec g2 = zeros<vec>(dim);
		unordered_map<unsigned, vec> tmp_Ww;		// HOGWILD! STYLE
		unordered_map<unsigned, vec> tmp_Ws;
		for (auto i : ques_indices){ 
			
			tmp_Ww[i] = Ww.col(i);
			f += tmp_Ww[i];
		} 
		for (auto i : answer_indices){
			
			tmp_Ws[i] = Ws.col(i);
			g1 += tmp_Ws[i];
		}
		for (auto i : wrong_answer_indices) {
			//if (i >= Ws.n_cols)cout << i << " "<< Ws.n_cols<< endl;
			tmp_Ws[i] = Ws.col(i);
			g2 += tmp_Ws[i];
		}
		double f_l1 = norm(f);
		double g1_l1 = norm(g1);
		double g2_l1 = norm(g2);
		double norm_dot_f_g1 = norm_dot(f, g1);
		double norm_dot_f_g2 = norm_dot(f, g2);
		double loss = gamma - norm_dot_f_g1 + norm_dot_f_g2;
		if (loss < 0) return 0;
		for (auto i : answer_indices){
			
			Ws.col(i) = tmp_Ws[i] + alpha * weight * (f / (f_l1 * g1_l1) - norm_dot_f_g1 / (g1_l1 * g1_l1) * g1);
		}
		for (auto i : ques_indices){
			
			//Ww.col(i) += alpha * wp;
			Ww.col(i) = tmp_Ww[i] + alpha * weight * (g1 / (f_l1 * g1_l1) - norm_dot_f_g1 / (f_l1 * f_l1) * f);
		}
		for (auto i : wrong_answer_indices){
			
			Ws.col(i) = tmp_Ws[i] - alpha * weight * (f / (f_l1 * g2_l1) - norm_dot_f_g2 / (g2_l1 * g2_l1) * g2);
		}
		for (auto i : ques_indices){
			
			Ww.col(i) = tmp_Ww[i] - alpha * weight * (g2 / (f_l1 * g2_l1) - norm_dot_f_g2 / (f_l1 * f_l1) * f);
		}
		return loss;
}
void QAEmbedding::subgraphPresentation(string &root, vector<unsigned> &indices){
	
	auto start_index = index_of_kb.size();
	double prob = min(1., 100.0 / graph[root].size());
	set<unsigned> subgraph_set;
	if (graph.count(root) == 0) cout << "can't find " << root << " in graph" << endl;
	for (auto & item : graph[root]){
		if (distribution(generator) <= prob){
			if (index_of_kb.count(item.first) == 0) cout << "11" << item.first << endl;
			if (index_of_kb[item.first] >= Ws.n_cols)cout << item.first << endl;
			
			subgraph_set.insert(index_of_kb[item.first] + start_index);
			for (auto & o : item.second){
				if (index_of_kb.count(o) == 0) cout << "12" << o << endl;
				if (index_of_kb[o] >= Ws.n_cols)cout << o << endl;
				
				subgraph_set.insert(index_of_kb[o] + start_index);
			}
		}
	}
	copy(subgraph_set.begin(), subgraph_set.end(), back_inserter(indices));
}
// Random select a negative data without go over the whole candidate sub-graph
void QAEmbedding::randomNegativeData(const string &topic, const string & correct_rel, vector<unsigned> &indices){
	
	// rough random
	
	
	// The probability of choice of 1-hop or 2-hop relation is determined by the portion of number of direct neighbor.
	bool is_two_hop = true;
	double one_hop_prob = n_direct_neighbor[topic] / n_neighbor_within_2[topic];
	if (distribution(generator) < one_hop_prob) is_two_hop = false;
	
	// Decide to choose 1-hop relation
	if (!is_two_hop){
		// random choose a relation which doesn't appear in correct path
		auto n_r1 = entity_to_relation_vec[topic].size();
		auto r1 = entity_to_relation_vec[topic][rand() % n_r1];
		
		if (r1 != correct_rel || n_r1 != 1) {
			while (r1 == correct_rel){
				r1 = entity_to_relation_vec[topic][rand() % n_r1];
			}
			auto &objs = graph[topic][r1];
			auto &o = objs[rand() % objs.size()];
			
			indices.push_back(index_of_kb[topic]);
			
			indices.push_back(index_of_kb[r1]);
			
			indices.push_back(index_of_kb[o]);
			subgraphPresentation(o, indices);
			return;
		}
		else{
			is_two_hop = true;
		}
		
	}
	if(is_two_hop){
		
		auto n_r1 = entity_to_relation_vec[topic].size();
		auto r1 = entity_to_relation_vec[topic][rand() % n_r1];
		// random choose a entity linking to topic as mediator entity
		auto m = graph[topic][r1][rand() % graph[topic][r1].size()];
		
		auto n_r2 = entity_to_relation_vec[m].size();
		// This is extreme case, the only 2-hop relation is correct path, any entity direct link to topic can be wrong answer
		if (n_r1 == 1 && graph[topic][r1].size() == 1 && n_r2 == 1 && correct_rel == r1 + " " + graph[m][entity_to_relation_vec[m][0]][0]){
			
			indices.push_back(index_of_kb[topic]);
			
			indices.push_back(index_of_kb[r1]);
			
			indices.push_back(index_of_kb[m]);
			subgraphPresentation(m, indices);
			return;
		}
		
		// If the selected mediator entity is a leaf node
		while (n_r2 == 0){
			// Only reconsidering mediator will lead to dead loop, because of an extreme case where all enitities linking topic through r1 are leaf node.
			// To simply code, we just rechoose the first relation. Don't worry that there is no 2-hop negative data, it guarantees that 2-hop must exists once code reachs here.
			r1 = entity_to_relation_vec[topic][rand() % n_r1];
			m = graph[topic][r1][rand() % graph[topic][r1].size()];
			n_r2 = entity_to_relation_vec[m].size();
		}
		auto r2 = entity_to_relation_vec[m][rand() % n_r2];
		while (r1 + " " + r2 == correct_rel ){
			if (n_r2 == 1){
				// Fallback to 1-hop path
				indices.push_back(index_of_kb[topic]);
				indices.push_back(index_of_kb[r1]);
				indices.push_back(index_of_kb[m]);
				subgraphPresentation(m, indices);
				return;
			}
			r2 = entity_to_relation_vec[m][rand() % n_r2];
		}
		auto &objs2 = graph[m][r2];
		auto &o = objs2[rand() % objs2.size()];
		
		indices.push_back(index_of_kb[topic]);
		indices.push_back(index_of_kb[r1]);
		indices.push_back(index_of_kb[r2]);
		indices.push_back(index_of_kb[o]);
		subgraphPresentation(o, indices);
		return;
	}
}

void QAEmbedding::train_one(Blob &blob, double &loss){
	
	auto &topic = blob.topicEntity();
	auto &correct_relation = blob.relation();
	//if(correct_relation!="") cout << correct_relation << endl;
	
	unsigned N = 0;
	unsigned Y = n_neighbor_within_2[topic]; // Number of entities that are reachable through 1-hop or 2-hop relation from topic
	unsigned n_correct_answers = 0;
	auto found = correct_relation.find(" ");

	if (found != string::npos){
		auto r1 = correct_relation.substr(0, found);
		auto r2 = correct_relation.substr(found + 1);
		if (graph[topic].count(r1) == 0) {
			//cout << "can't find " << r1 << " in graph[" << topic << "] " << correct_relation << endl;
			return;
		}
		auto &objs = graph[topic][r1];
		for (auto &o : objs){
			if (graph[o].count(r2) == 0)continue;
			n_correct_answers  += graph[o][r2].size();  // Statistic number of correct answers
		}
	}
	else{
		if (graph[topic].count(correct_relation) == 0){
			//cout << "2333 can't find " << correct_relation << " in graph[" << topic << "] " << correct_relation << "ddd" << endl;
			return;
		}
		n_correct_answers += graph[topic][correct_relation].size();
	}
	while (true){
		// Pick a random negative data
		vector <unsigned> wa_indices;
		randomNegativeData(topic, correct_relation, wa_indices);
		N += 1;
		unsigned rank = unsigned(floor((Y - n_correct_answers) / N));
		if (rank >= MAX_TABLE_SIZE) rank = MAX_TABLE_SIZE - 1;
		double weight = l_table[rank];
		double loss_one = gradientDescent(blob.questionIndices(), blob.answerIndices(), wa_indices, weight);

		if (loss_one > 0.){
			loss += loss_one;
			break;
		}
		if (N > Y - n_correct_answers)break;
	}
}
//void QAEmbedding::train_one(Blob &blob, double &loss){
//	auto &q_entity = blob.topicEntity();
//
//	auto &correct_relation = blob.relation();
//	auto start_index = index_of_kb.size();
//	static uniform_real_distribution<double> distribution(0.0, 1.0);
//	static default_random_engine generator;
//	/*int neighbor_count_1_hop = 0;
//	int neighbor_count_2_hop = 1;*/
//	for (auto &item : graph[q_entity]){
//		
//		// 1-hop negative data
//		if (item.first != correct_relation) {
//			//neighbor_count_1_hop += item.second.size();
//			for (auto &wa : item.second){
//			
//				// sampling 50 % of time from the set of entities connected to the entity of the question 
//				if (distribution(generator) <= 0.5){
//
//					vector <unsigned> wa_indices;
//					wa_indices.push_back(index_of_kb[q_entity]);
//					wa_indices.push_back(index_of_kb[item.first]);
//					wa_indices.push_back(index_of_kb[wa]);
//					// use at most 100 paths in the subgraph representation
//					if (graph.count(wa) > 0){
//						auto total_path_num = graph[wa].size();
//						double prob = min(1., 100.0 / total_path_num);
//						// add subgraph
//						set<unsigned> subgraph_set;
//						for (auto & item2 : graph[wa]){
//							if (distribution(generator) <= prob){
//								subgraph_set.insert(index_of_kb[item2.first] + start_index);
//								for (auto & o : item2.second){
//									subgraph_set.insert(index_of_kb[o] + start_index);
//								}
//							}
//						}
//						copy(subgraph_set.begin(), subgraph_set.end(), back_inserter(wa_indices));
//					}
//					loss += gradientDescent(blob.questionIndices(), blob.answerIndices(), wa_indices);
//				}
//			}
//		}
//		// 2-hop negative data
//		if (distribution(generator) > 0.5) continue;
//		auto mediate = item.second[rand() % item.second.size()];
//		if (graph.count(mediate) > 0) {
//			for (auto &item2 : graph[mediate]){
//				if (distribution(generator) > 0.5) continue;
//				if (item.first + " " + item2.first == correct_relation) continue;
//				//neighbor_count_2_hop += item2.second.size();
//				auto wa = item2.second[rand() % item2.second.size()];
//				
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
//				loss += gradientDescent(blob.questionIndices(), blob.answerIndices(), wa_indices);
//				
//			}
//		}
//		
//		//printf("neighbor_count_1_hop = %d neighbor_count_2_hop = %d\n", neighbor_count_1_hop, neighbor_count_2_hop);
//		
//	}
//}
void QAEmbedding::train(int n_epoch, double _alpha, double _gamma){
	alpha = _alpha;
	gamma = _gamma;
	unsigned long n_workers = 32;
	
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
	clock_t time_begin = clock();
	for (int epoch_index = 1; epoch_index <= n_epoch; ++epoch_index){
		
		vector <double> loss_s(n_workers, 0);
		auto block_start = blob_vec.begin();

		for (unsigned long i = 0; i < n_workers - 1; ++i){
			auto block_end = block_start;
			std::advance(block_end, block_size);

			threads[i] = thread([&, block_start, block_end, i](){
				for (auto it = block_start; it != block_end; ++it)
					train_one(*it, loss_s[i]);
			});

			block_start = block_end;
		}
		for (auto it = block_start; it != blob_vec.end(); ++it){
			train_one(*it, loss_s[n_workers - 1]);
		}
		for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
		total_loss = std::accumulate(loss_s.begin(), loss_s.end(), 0);

		printf("# %d epoch, loss = %f, it takes %f s\n", epoch_index, total_loss, double(clock() - time_begin) / CLOCKS_PER_SEC);
		save();
		fin.close();
		time_begin = clock();
	
	}

}
int main(int argc, char **argv){
	auto qae = QAEmbedding(64, "data/qa.train", "data/result.matrix", "data/word.list", "data/fb.entry.list", "data/fb.triple");
	qae.train(100, 0.001, 0.1);
	qae.save();
}
