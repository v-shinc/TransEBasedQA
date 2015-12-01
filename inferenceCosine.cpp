#define ARMA_DONT_USE_CXX11
#include <armadillo>
#include <string>
#include <unordered_map>
#include <boost/algorithm/string.hpp>
#include <queue>
#include <vector>
#include <fstream>
#include <tuple>
#include <random>
#include <algorithm>
#include <thread>
#include <unordered_set>
#include <atomic>

//#include <boost/lexical_cast.hpp>
using namespace arma;
using namespace std;

using map_type = unordered_map < string, unsigned > ;
using graph_type = unordered_map < string, unordered_map<string, vector<string>> > ;

using triple_type = tuple < double, string, string>;
double scoreFunc(vec &f_q, vector<unsigned> &a_indices, mat &Ws){
	vec g_a = zeros<vec>(f_q.n_rows);
	for (auto i : a_indices) {
		g_a += Ws.col(i);
	}
	return norm_dot(g_a, f_q);
}
struct Cmp{
	bool operator()(const triple_type &lhs, const triple_type &rhs){
		return get<0>(lhs) > get<0>(rhs);
	}
};
struct AnswerInfo{
	string answer;
	string topic_entity;
	int n_hop;
};
void beam_search_c2(graph_type& graph, const string &q_entity, vec &f_q, map_type &index_of_kb, mat & Ws, vector<pair<double, AnswerInfo>> &answers){

	priority_queue<triple_type, vector<triple_type>, Cmp>  pq;
	static uniform_real_distribution<double> distribution(0.0, 1.0);
	static default_random_engine generator;
	if (index_of_kb.count(q_entity) == 0){
		cout << "key error: can't find " << q_entity << "in index_of_kb" << endl;
	}
	
	for (auto &item1 : graph[q_entity]){
		
		auto rel1 = item1.first;
		for (auto m : item1.second){ // mediator entity
			for (auto item2 : graph[m]){
				auto rel2 = item2.first; // the 2nd hop relation
				vector<unsigned> answer_vec;
				answer_vec.push_back(index_of_kb[q_entity]);
				answer_vec.push_back(index_of_kb[rel1]);
				answer_vec.push_back(index_of_kb[rel2]);
				double score = scoreFunc(f_q, answer_vec, Ws);
				if (pq.empty() < 10){
					pq.push(std::move(make_tuple(score, rel1, rel2)));
				}
				else if (get<0>(pq.top()) < score){
					pq.pop();
					pq.push(std::move(make_tuple(score, rel1, rel2)));
				}
			}
		}
	}
	unsigned start_index = index_of_kb.size();
	while (!pq.empty()){
		auto e = pq.top();
		pq.pop();
		string rel1 = get<1>(e);
		string rel2 = get<2>(e);

		for (auto & meditor_node : graph[q_entity][rel1]){
			
			for (auto answer_node : graph[meditor_node][rel2]){
				vector<unsigned> answer_vec;
				answer_vec.push_back(index_of_kb[q_entity]);
				answer_vec.push_back(index_of_kb[rel1]);
				answer_vec.push_back(index_of_kb[rel2]);
				answer_vec.push_back(index_of_kb[answer_node]);
				// add subgraph
				unordered_set<unsigned> subgraph_set;
				for (auto & item : graph[answer_node]){
					double prob = min(1.0, 100.0 / graph[answer_node].size());
					if (distribution(generator) <= prob){
						subgraph_set.insert(start_index + index_of_kb[item.first]); // relation linking to answer node
						for (auto & o : item.second)
							subgraph_set.insert(start_index + index_of_kb[o]);
					}
				}
				//cout << "Element number of subgraph = " << subgraph_set.size() << endl;
				copy(subgraph_set.begin(), subgraph_set.end(), back_inserter(answer_vec));
				auto score = scoreFunc(f_q, answer_vec, Ws);
				answers.push_back(std::move(make_pair(score, std::move(AnswerInfo{ answer_node, q_entity, 2}))));
			}
		}
	}
	//sort(answers.begin(), answers.end(), [](const pair<double, string> &lhs, const pair<double, string>&rhs){return lhs.first > rhs.first; });
}
void strategy_c1(graph_type& graph, const string &q_entity, vec &f_q, map_type &index_of_kb, mat & Ws, vector<pair<double, AnswerInfo>> &answers){

	if (index_of_kb.count(q_entity) == 0){
		cout << "key error: can't find " << q_entity << "in index_of_kb" << endl;
	}
	
	static uniform_real_distribution<double> distribution(0.0, 1.0);
	static default_random_engine generator;
	unsigned start_index = index_of_kb.size();
	for (auto &item1 : graph[q_entity]){
		for (auto &candidate : item1.second){
			vector<unsigned> answer_vec;
			answer_vec.push_back(index_of_kb[q_entity]);
			answer_vec.push_back(index_of_kb[item1.first]);  // add relation in path
			answer_vec.push_back(index_of_kb[candidate]);

			double prob = min(1.0, 100.0 / graph[candidate].size());
			unordered_set<unsigned> subgraph_set;
			// add subgraph
			for (auto & item2 : graph[candidate]){
				if (distribution(generator) <= prob){
					subgraph_set.insert(start_index + index_of_kb[item2.first]);
					for (auto & obj : item2.second){
						subgraph_set.insert(start_index + index_of_kb[obj]);
					}
				}
			}
			//cout << "Element number of subgraph = " << subgraph_set.size() << endl;
			copy(subgraph_set.begin(), subgraph_set.end(), back_inserter(answer_vec));
			auto score = scoreFunc(f_q, answer_vec, Ws);
			answers.push_back(std::move(make_pair(score * 1.5, std::move(AnswerInfo{ candidate, q_entity, 1 }))));
			
		}
	}
	//sort(answers.begin(), answers.end(), [](const pair<double, string> &lhs, const pair<double, string>&rhs){return lhs.first > rhs.first; });
}
struct Blob{
	string question;					// question
	vector<string> topic_entities;		// entities appearing in question
	string gold_answers_str;			// standard answers	which can be aligned to freebase
	string original_size_of_gold;		// number of standard answers. Those answers which fail to be aligned is included
	string predicated;
};
void inference_one(Blob &blob, 
	unordered_map<string, unsigned> &index_of_word,
	unordered_map<string, unsigned> &index_of_kb,
	unordered_map <string, unordered_map<string, vector<string>>> &graph,
	mat &Ws, mat &Ww){
	
	// Transform each word in question to corresponding index
	vector<string> words;
	
	vec f_q = zeros<vec>(Ww.n_rows);
	boost::split(words, blob.question, boost::is_any_of(" "));
	for (auto &w : words)
	{
		if (index_of_word.count(w) > 0){
			auto i = index_of_word[w];
			f_q += Ww.col(i);
		}
		else{
			cout << "cannot find " << w << " in index_of_word\n";
		}
	}
	
	vector<pair<double, AnswerInfo>>  answers;
	// Go over all candidate question topic entities
	for (auto &topic_e : blob.topic_entities){
		strategy_c1(graph, topic_e, f_q, index_of_kb, Ws, answers);
		//beam_search_c2(graph, topic_e, f_q, index_of_kb, Ws, answers);
	}
	sort(answers.begin(), answers.end(), [](const pair<double, AnswerInfo> &lhs, const pair<double, AnswerInfo>&rhs){return lhs.first > rhs.first; });

	std::ostringstream os;
	unordered_set<string> appeared;
	auto highest_score = answers[0].first;

	double threshold = 0.1;
	for (auto &a : answers){
		AnswerInfo &info = a.second;
		// The candidates whose scores are not far from the best answer are regarded as predicated results.
		// The threshould is set to be same with the margin defined at training stage.
		//if (highest_score - threshold > a.first) break;
		if (appeared.count(a.second.answer) == 0){
			os << info.answer << ":" << a.first << ":" << info.topic_entity << ":" << info.n_hop << " "; 
			appeared.insert(info.answer);
		}
	}
	string answer_str = os.str(); // Extra space at last need to be removed
	if (answer_str.back() == ' ')
		answer_str.pop_back();
	//static std::atomic<int> lno(0);
	//Ignore thread collision
	static int lno = 0;
	os.str("");
	//os.clear();
	os << blob.question << "\t" << blob.gold_answers_str << "\t" << answer_str << "\t" << blob.original_size_of_gold;
	blob.predicated = os.str();
	lno++;
	cout << "Process to line " << lno << endl;
}
/* 
fb_path: path of freebase
word_list_path: file path of list of words which appear in questions
kb_list_path: file path of relations and entities list
test_data_path: webquestion test file
matrix_dir: directory for well-trained matrix. The matrix is saved by armadillo's save function,
			word matrix file is named wordEmbedding_s.out, matrix for entity and relation is stored in kbEmbedding_s.out file
*/
void inference(const string&fb_path, const string& word_list_path, const string& kb_list_path, const string&test_data_path, const string& matrix_dir = "data/result.matrix"){

	mat Ww, Ws;
	Ww.load(matrix_dir + "/wordEmbedding_s.out");
	Ws.load(matrix_dir + "/kbEmbedding_s.out");
	int dim = Ws.n_rows;
	cout << "column number of Ww " << Ww.n_cols << "row number of Ww "<< Ww.n_rows << endl;
	cout << "column number of Ws " << Ws.n_cols << "row number of Ws "<< Ws.n_rows << endl;
	unordered_map<string, unsigned> index_of_word;
	unordered_map<string, unsigned> index_of_kb;
	unordered_map <string, unordered_map<string, vector<string>>> graph;
	string sline;

	unsigned no = 0;
	fstream word_fin(word_list_path);
	while (getline(word_fin, sline)) index_of_word[sline] = no++;
	word_fin.close();
	cout << "size of index_of_word" << no << " "<<index_of_word.size()<<endl;

	unsigned kb_no = 0;
	fstream kb_fin(kb_list_path);
	while (getline(kb_fin, sline))index_of_kb[sline] = kb_no++;
	kb_fin.close();
	cout << "size of index_of_kb" << kb_no << " "<<index_of_kb.size()<<endl;

	// Load freebase as graph
	fstream fb_fin(fb_path);
	while (getline(fb_fin, sline)){
		vector<string> strs;
		boost::split(strs, sline, boost::is_any_of("\t"));
		graph[strs[0]][strs[1]].push_back(strs[2]);
	}
	fb_fin.close();

	int lno = 0;
	char out_path[] = "data/inference.result";
	FILE *fout = fopen(out_path, "w");
	fstream fs(test_data_path);
	
	
	/* the input format is 
		(question (word-version), candidate topic entities list , gold answer set, original size of gold answer set), columns are separated by tab, 
		because some answer can't aligned to freebase, gold answer set only contains the aligned answer, the original size of answer set is not consist
		with the size of gold answer set.
	*/
	// Load test file and store it 
	vector<Blob> test_data;
	while (std::getline(fs, sline)){
		vector<string> strs;
		boost::split(strs, sline, boost::is_any_of("\t"));
		Blob blob{
			strs[0],
			vector<string>(),
			strs[2],
			strs[3],
			""
		};
		boost::split(blob.topic_entities, strs[1], boost::is_any_of(" "));
		test_data.push_back(std::move(blob));
	}
	cout << "Start to inference\n";
	// Create threads and assign the work
	int n_workers = 16;
	vector<thread> threads(n_workers - 1);
	unsigned long block_size = test_data.size() / n_workers;

	auto block_start = test_data.begin();
	for (unsigned long i = 0; i < n_workers - 1; ++i){
		auto block_end = block_start;
		std::advance(block_end, block_size);

		threads[i] = thread([&index_of_kb, &index_of_word, &graph, &Ws, &Ww, block_start, block_end, i](){
			
			for (auto it = block_start; it != block_end; ++it){
				inference_one(*it, index_of_word, index_of_kb, graph, Ws, Ww);
			}
		});

		block_start = block_end;
	}
	for (auto it = block_start; it != test_data.end(); ++it){
		inference_one(*it, index_of_word, index_of_kb, graph, Ws, Ww);
	}
	for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));
	for_each(test_data.begin(), test_data.end(), [&fout](Blob &b){fprintf(fout, "%s\n", b.predicated.c_str()); });
	fs.close();
}
int main(int argc, char** argv){
	inference("data/fb.triple", "data/word.list", "data/fb.entry.list", "data/webquestions.test.qa");
}