/*
	Programmer:	Adam Mischke
	Professor:	Joshua L. Phillips
	Program:	Supervised Learning - Project 3
	Started:	Friday,	Nov. 3 at 16:53:00
	Date Due:	Tuesday, Nov. 14 by 10:00pm
	Date Finished: Monday, Nov. 13 at 17:30:00
	Header file for id3.cpp
*/

#ifndef ID3_H
#define ID3_H

#include <vector>
#include <set>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <numeric>
#include <cmath>

using std::vector;
using std::cout;
using std::string;
using std::stringstream;
using std::getline;
using std::cin;
using std::ifstream;
using std::endl;
using std::ostream;
using std::pair;
using std::set;
using std::find;

// split struct. probably should have just used pair
struct split{

	int position;
	double avg;

};

// tree node that we use to build a binary tree
struct Node{

	int splitColumn;		// column where we split
	double split;			// average split value

	int classifier;			// name of the classified object
	bool isTerminal;		// flag on whether this is a leaf or not..

	Node* leftChild;
	Node* rightChild;

};

/* typedeffing for ease of use */
typedef vector<double> vd;
typedef vector<vd> vvd;
typedef vector<int> vi;
typedef vector<vector<int > > vvi;
typedef pair<double, double> pdd;

/* Printing vectors */
template <typename t> void print(vector<vector< t > >  v);
template <typename t> void print(vector< t >   v);

/* printing pairs and splits */
ostream& operator<<(ostream& o, const split& s){
	return o << "pos: " << s.position << " avg: " << s.avg << "\n";
}
ostream& operator<<(ostream& o, const pair<double,double>& p){
	return o << "first: " << p.first << " second: " << p.second << "\n";
}


/* FILE INPUT HANDLERS */
void handleTestIn(vvd& v, const string in);
void handleFileIn(vvd& data, const string train_fn);


// Attribute sorting from Dr. Phillips
vector<vector<int> > sortAttributes(vvd data)
{
  vector<vector<int> > indices;
  vd *ptr;
  indices.resize(data.size());
  for (int x = 0; x < indices.size(); x++) {
    indices[x].resize(data[x].size());
    iota(indices[x].begin(),indices[x].end(),0);
    ptr = &(data[x]);
    sort(indices[x].begin(),indices[x].end(),
	 [&](size_t i, size_t j){ return (*ptr)[i] < (*ptr)[j]; });
  }
  return indices;
}

// calculates the shannon entropy
double E(vd high, vd low, double high_, double low_ );

// finds the probability of the class column in the data
vd findProbClassification(const vvd& data);

// calculates information gain from a row
double I(const vvd& data, const vd& vec);

// calculates total information gain
double Gain(double i, double e){return(i-e);}

// splits from an attribute
void splitFromAttr(double last, double next, int* num, vector<split>& sVec, 
					const vd& iVec, int j);

// handles the split loop
int handleSplits(int k,  const vector<split>& sVec,const vd& iVec,
				const vvd& data, const vvi& indices, pdd& gain_and_pos);

// finds the max gain from a split
pdd findMaxGainFromSplit(const vvd& data, const vvi& indices, split& highSplit);

// finds all the information about the classification column
int findAllClassifications(const vvd& data, const vvi& indices);

// prunes the table after a split to their repective children
vvd pruneTable(const vvd& data, vvi& indices, vvi &outIndices, 
			  int col, double value, int row, string side);

/* BUILDING THE TREE */
Node* decisionTreeLearning(Node* root, vvd& data, vvi& indices);

/* DELETING THE TREE */;
void deleteTree ( Node* root );

/* TESTING THE TREE */
int testLine(vd v, Node* root);
int testTree(Node* root, const vvd& v);


#endif