/*
	Programmer:	Adam Mischke
	Professor:	Joshua L. Phillips
	Program:	Supervised Learning - Project 3
	Started:	Friday,	Nov. 3 at 16:53:00
	Date Due:	Tuesday, Nov. 14 by 10:00pm
	Date Finished: Monday, Nov. 13 at 17:30:00

	Description:
		This program uses an ID3 decision tree from labeled
		classification data provided by iris-data.txt to classify
		testing data.
	
	Argv:
		integer: the number of real-valued features in the data set
		string: input training data filename
		string: input testing data filename

	ID3:

	In decision tree learning, ID3 (Iterative Dichotomiser 3) 
	is an algorithm invented by Ross Quinlan[1] used to generate 
	a decision tree from a dataset. ID3 is the precursor to the 
	C4.5 algorithm, and is typically used in the machine learning 
	and natural language processing domains.

		function decision-tree-learning (examples, attributes,  default)
			begin
				if empty(examples) then return (default)
			     else if all examples have the same classification then return the classification
			     else if attributes is empty then return majority-classification(examples)
			     else
					best-attr choose-attribute(attributes, examples)
				     	tree  a new decision tree with root test best-attr
			     		for each value v of best-attr do
			     		begin
			           		v-examples  subset of examples with best-attr = v
						subtree  decision-tree-learning (v-examples, 
									attribute – best-attr, 
									examples)
						add a branch to tree with arc labeled v and subtree
			 		end
					return (tree)
			end

	
	Information Gain:
		I(p,n) = -p/(p+n) log2p/(p+n) - n/(p+n) log2n/(p+n)

	Shannon Entropy:
		E(A) = ∑(i=1,v) (pi + ni)/(p + n) I(pi, ni)

	Gain minimize E(A):
		gain(A) = I(p,n) - E(A)
		

	Cross Validaton Style:
		Partition data into N disjoint sets S= {S1, S2, … SN}
		i=1
			loop N times:
			Let training set be (S – Si), and
		  	      test set be Si,
			Learn the classifier based on the current training set,
			Test the performance of the classifier on the current test set
			Record the predication accuracy
			i = i + 1;
		end  loop
		Compute the average predication accuracy for the N runs

*/


#include "id3.hpp"

int FEATURE_SIZE;	// size of the features
int class_size=3;	// how many distinct classifications

int main(int argc, char* argv[])
{

	FEATURE_SIZE = atoi(argv[1]); // not used
	string trainFn = argv[2]; 
	string testFn = argv[3];

	// test files w/o argv
	//trainFn = "temp.32476.train.txt";
	//testFn = "temp.32476.test.txt";

	// 2D vector of doubles that will hold all of the data from the training file
	vvd data;

	// 2D vector of ints that will hold all of the indices for the sorted data
 	vvi indices;

 	// 2D vector of doubles that will hold all of the test data
 	vvd testData;

 	// number of test data values that were classified correctly
 	int numRight;

 	// root of the ID3 tree
 	Node* root = new Node;
 	root->isTerminal = false;

 	// data <= trainFn
  	handleFileIn(data,trainFn);

  	FEATURE_SIZE = data.size();
	// sorts the attributes by column 
	indices = sortAttributes(data);


	//attr_size = data.size();	// how many columns of attributes we have
	//int data_size = data[0].size();	// how many rows of data we have

	//cout << "attr size: " << attr_size-1 << " data size: " << data_size << "\n";

	// assigns the root of our ID3 tree
	// recursively calls itself from the data and
	// the sorted indices
	root = decisionTreeLearning(root, data, indices);


	//cout << "done!\nNodes: " << test << "\n";
	//cout << "# of leaves: " << terminal << "\n";

	// testData <= test file
	handleTestIn(testData,testFn);

	//cout << data[0].size() << " " << testData.size();
	//exit(0);

	numRight = testTree(root, testData);

	//cout << "Number Right: " << numRight << "\n";
	cout << numRight << "\n";

	deleteTree(root);
	//cout << "tree deleted!\n";
	return 0;
	
}

/* Templated Print Statements for double and single vectors */
template <typename t>
void print(vector<vector< t > >  v){
	for (auto& i : v){
		for (auto& j : i){	
			cout << j << " ";
		}
		cout << "\n";
	}cout << "\n";

}
template <typename t>
void print(vector< t >   v){
	for (auto& i : v){
		cout << i << " ";
	}
	cout << "\n";
}


/* User defined functions for main */

// calculates and returns the Information gain of the classifications
// value from data from a row
double I(const vd& vec)
{
	int size = vec.size();		// find the size of the vector of classifications
	double prob = 0;			// probability counter
	double total = 0;			// total probability

	for (int i=0; i<class_size; i++){
		// count how many classifications there are
		prob = count(vec.begin(), vec.end(), i) / (double) size;
		//cout << "prob: " << prob << "\n";

		if (prob != 0)		// not -inf
		{
			total += prob * (-log2(prob));	// add to the total
		}
	}
	return total;
}

// E(size)= ­ -P(big)*[P(+|big)*log_2(P(+|big))+P(­-|big)*log_2(P(­-|big))] 
//– P(small)*[P(+|small)*log_2(P(+|small))+P(­-|small)*log_2(P(­-|small))]
// dynammically calculates and returns E
double E(vd high, vd low, double high_, double low_ )
{	
	// two temp values for low and high
	double temp1=0;
	double temp2=0;

	// iterate through the classes
	for (int i=0; i<class_size-1; i++){
		// if high probability isn't 0,
		if (high[i] != 0)
			temp1 += (high[i] * log2(high[i]));	// add to the high
		//cout << temp1 << " + ";
	}
	temp1 *= -high_;	// add the last part
	//cout << "E " << temp1 << "\n";

	// iterate through the classes
	for (int i=0; i<class_size-1; i++){
		// if low probability isn't 0,
		if (low[i] != 0)
			temp2 += (low[i] * log2(low[i]));	// add to the low
			//cout << temp2 << " + ";
	}
	temp2 *= -low_;	// add last part
	//cout << "E " << temp2 << "\n";

	return temp1 + temp2; // return sum
}

// find probability finds the probability of a 2D vector
// returns the probabilities in a vector of doubles
vd findProbClassification(const vvd& data)
{	
	// itialize some values
	int temp, count=0;
	// initialize a vector to send back
	vd tempVec(class_size);

	int attrSize = data[0].size() - 1;	// last column of attributes is the classification

	// iterate through the rows
	for (int i=0; i<data.size(); i++){
		
		temp = data[i][attrSize]; 	// find the classification of the row
		tempVec[temp] +=1;			// add a one to the temp column
		count++;					// increase total count

	}
	
	for (auto& v : tempVec){		// iterate through the temp vec,
		v /= (double) count;		// update the probability in the vectors

	}
	return tempVec;		// return the vector with the proper probabilites per classification
}


// creates and references a split from an attribute
// updates num and split vector
void splitFromAttr(double last, double next, int* num, 
				   vector<split>& sVec, const vd& iVec, int j)
{
      	// find the average of the split
  		double avg = (next + last) / 2;
  		//cout << " checked! avg: " << avg << " i: " << iVec[j];

  		// create a split
  		split s;
  		s.position = j;	// position in the row
  		s.avg = avg;	// average of the split

  		// put the split into a vector for splites
  		sVec[*num] = s;

  		// increase the counter of splits
  		(*num)++;
}

int handleSplits(int k, const vector<split>& sVec, const vd& iVec, const vvd& data, 
	             const vvi& indices, pdd& gainAndPos)
{

	// create an I(x) value from our classification column per permutation
	double iVal = I(iVec);

	// display splites
    //cout << "splits:\n";
    
    // initialize the gain and position
    gainAndPos = {0,0};
    
    vd lowProbabilityClassification;	// vector of low probability classifications
    vd highProbabilityClassification;	// "" but for high

    int splitCount=0;
    int splitPos=0;
    // iterate over the splits in the splits vector
    for (auto& s : sVec){ 
    	//cout << s;		// print the split vector

    	vvd highRow;	// 2D highrow
		vvd lowRow;		// 2D lowrow
	    // rows sorting by each column smallests to greatest
	    for (int j = 0; j < data[0].size(); j++) {

	    	// create a new row
	    	vd row;

	    	// the column
	      	for (int i = 0; i < data.size(); i++) {
	      		// push the data into the row
	      		row.push_back(data[i][indices[k][j]]);
	      	}
	      	// if the split average is greater than the row we're looking at,
	      	if(s.avg > row[k] )
	      	{	
	      		lowRow.push_back(row);	// push the rows into the low row
	      	}
	      	// the split must be greater
	      	else
	      	{
	      		highRow.push_back(row); // so push these rows to the high row
	      	}
	    }
	    	double gain;
	    	double lowSize;
	    	double highSize;
	    	double lowProb;
	    	double highProb;
	    	double e;

	    	// find the sizes of the vectors
		   	lowSize = lowRow.size();
		    highSize = highRow.size();

		    // finds the low probability and high probability
		    lowProb = lowSize / (lowSize + highSize);
		    highProb = 1 - lowProb;
		    //cout << "low prob: " << low_prob <<  " high prob: " << high_prob << "\n";

		    // finds the probability of the classifications based on the permutation
		    lowProbabilityClassification = findProbClassification(lowRow);
		    highProbabilityClassification = findProbClassification(highRow);

		    // find E
		    e = E(highProbabilityClassification, lowProbabilityClassification, 
		    		     highProb, lowProb);
		    //cout << "E: " << e << "\n";

		    // calculate gain
		    gain = Gain(iVal,e);
		    //cout << "Gain: " << gain << "\n";

		    // if this gain is higher than the gain in the splits before
		    if(gain > gainAndPos.first)
		    {
		    	gainAndPos.first = gain;	// set this as the top gain
		    	gainAndPos.second = k;		// set the permutation as the position
		    	splitPos = splitCount;				// row of the split with the highest gain
		    }	
		    splitCount++;							
    }

    //cout << "top split per permutation " << k << " " << topSplit;
    return splitPos;	// returns the row indice of the highest split
}

// Dr. Phillip's file input function
void handleFileIn(vvd& data, const string train_fn)
{
	double value;
	string line;
	// simulates a 3 dimensional vector, bur returned in order by column
  	// iterate with a k permutation
  	ifstream train;
  	train.open(train_fn);			// open the training file
  	
	getline(train,line);			// get lines into a string stream
	stringstream parsed(line);		// named parsed per line
	  
	// Prep vectors...
	while (!parsed.eof()) {			
	    parsed >> value;					// parse
	    data.push_back(vd());	// prep 2D vector data
	}
	  
	while (!train.eof()) {					
	    stringstream parsed(line);
	    for (int i = 0; i < data.size(); i++) {
	      parsed >> value;			// parse
	      data[i].push_back(value); // push into the vector
	    }
	    getline(train,line);	// get line
	}
	train.close();	// close the file
}


// finds the max gain from data, indicies and updates a highsplit value
// returns the max gain pair
pdd findMaxGainFromSplit(const vvd& data, const vvi& indices, split& highSplit)
{
	// initialize a next and last state attributes
	double lastAttribute=-1, nextAttribute;

	// was data_size
	vd iVec(data[0].size());		// information vector for the cases

	pdd gainAndPos(0,0);			// temp pair
	pdd maxGain(0,0);				// initialize a pair (gain, which column won)

	int numOfPermutations = data.size()-1;  // (4)
	int numOfRows = data[0].size();			// 

	// Apply permutation for specific column
	for (int k = 0; k < numOfPermutations+1; k++) {

		vector<split> sVec(numOfRows);	// vector of split information
		int numSplits=0;				// number of splits we have per permutation

		//cout << "permutation: " << k << "\n";
	    // rows sorting by each column smallests to greatest
	    for (int j = 0; j < numOfRows; j++) {
	    	
	    	/*
	    	// debug printing
	      	for (int i = 0; i < data.size(); i++) {	
				cout << data[i][indices[k][j]] << " ";
	      	}
	      	*/

	      	// add the classificaction per row into a vector to calculate I(x)
	      	iVec[j] = data[numOfPermutations][indices[k][j]];

	      	// grab the attribute and print
	      	nextAttribute = data[k][indices[k][j]];

	      	// if this isn't the first attribute and the next attribute is greater than the last
	      	if (lastAttribute != -1 && nextAttribute > lastAttribute)
	      	{		
	      		// split from attribute
	      		splitFromAttr(lastAttribute,nextAttribute,&numSplits,sVec,iVec,j);
	      	}

	      	// make the last attribute the one we just used
	      	lastAttribute = nextAttribute;
	    	//cout << endl;
	    }	

		sVec.resize(numSplits);

	    // the works, returns the indice where the highest split is
	    int p = handleSplits(k,sVec, iVec, data, indices, gainAndPos);
	
	    //cout << "Max Gain: " << gainAndPos << "\n";

	    // if the gain avg is greater than the max gain that we have (first)
	    // break ties by lesser classification number (second)
	    if(gainAndPos.first > maxGain.first && maxGain.second <= gainAndPos.second)
	    {
	    	maxGain = gainAndPos;	// set max gain  (gain, classification)
	    	highSplit = sVec[p];	// set highsplit (avg, row where the split occurs)
	    }

	}
    //cout << "Acutal max gain: " << maxGain;
    //cout << "Actual max split: " << highSplit;
    return maxGain;	// return max gain
}

// finds if all the classifications in the 2D vector are the same
// if it is, it returns the classification
// if not, it returns -1
int findAllClassifications(const vvd& data){

	bool first = true;	// flag for the first value
	int value = -1;		// initial value
	// the column where our classifications in the vector are
	int classColumn = data.size()-1;

	// iterate through the rows..
	for (auto& c : data[classColumn]){
		if (first)
		{
			value = c;		// set the first value,	
			first=false;	// flag off
		}
		if(value != c)		// if c isn't the same
		{
			value = -1;		// wrong value flag
			break;			// break
		}
	}
	return value;			// return value
}

// returns a pruned table and new set of indices
vvd pruneTable(const vvd &data, vvi &indices, vvi &outIndices, 
		       int col, double value, int row, string side)
{	
	// iterators to build the correct table
	int tableSize=0,dataColumn=0;

	// number of columns
	int columnSize = data.size();

	// new table that we'll make
	vvd prunedTable(columnSize, vd (row));

	// header row for building the 2D vector
	vd headerRow;

	// if this is the low
	if (side == "low")
	{	
		// iterate through the data vector based on the column
		for (auto& c : data[col]){
			// if the number in the column is less than
			if (c < value)
			{	
				// add all of the values into their respective places
				for (int i = 0; i < columnSize; i++) {
					prunedTable[i][tableSize] = data[i][dataColumn];
  				}
  				// increase the table size
				tableSize++;
			}
			// increase the column number
			dataColumn++;
		}
	}
	else if (side == "high")
	{
		for (auto& c : data[col]){
			if (c >= value)
			{	
				// add all of the values into their respective places
				for (int i = 0; i < columnSize; i++) {
					//cout << data[i][j] << " ";
					prunedTable[i][tableSize] = data[i][dataColumn];
  				}
  				// increase the table size
				tableSize++;
  				//cout << "\n";
			}
			// increase the column number
			dataColumn++;
		}
	}
	// resort for the attributes
	outIndices = sortAttributes(prunedTable);
	return prunedTable;	// return the pruned table
}

// recursively builds the ID3 tree
Node* decisionTreeLearning(Node* root, vvd& data, vvi& indices)
{	
	// the classification, if all in data are the same number
	int c;

	//if(test >15) return nullptr;
	//if ( data[0].empty() ) exit(0);

	// if the all of the classifications in this data set are the same,
	if ( (c = findAllClassifications(data) ) != -1)
	{	
		root->isTerminal = true;	
		root->leftChild=nullptr;	// no children on a terminal
		root->rightChild=nullptr;
		root->classifier = c;		// classifier is the return value from above
		return root;				// return this node
	}
	else
	{
		// initialize some vectors for a left node and right node
		vvd leftTable;
		vvd rightTable;
		vvi leftIndices;
		vvi rightIndices;

		// create a new split value
		// average split value and position in the row that it is
		split highSplit = {0};

		// first = gain, second = column of where the gain was found
		pdd gain = findMaxGainFromSplit(data, indices, highSplit);

		Node* leftC = (Node*) new Node;			// create new nodes
		Node* rightC = (Node*) new Node;

		root->isTerminal = false;				// this is not a terminal
		root->splitColumn = gain.second;		// column where we split
		root->split = highSplit.avg;			// where the split is

		//cout << "highsplit:\nAVG: " << highSplit.avg << " ROW: " << highSplit.position << "\n";
		//cout << "GAIN: " << gain.first << " COLUMN " << gain.second <<  "\n";
		
		// prune the left table
		leftTable = pruneTable(data, indices, leftIndices, gain.second, highSplit.avg, highSplit.position,"low");
		
		//cout << data.size();
		//exit(0);
		// prune the right table
		rightTable = pruneTable(data, indices, rightIndices, gain.second, highSplit.avg, data[0].size()-highSplit.position,"high");
		//cout << "done pruning.. \n";

		// print out the data and the pruned split tables
		/*
		cout << "data table size: " << data[0].size() << "\n";
		print(data);
		cout << "left table size: " << leftTable[0].size() << "\n";
		print(leftTable);
		cout << "right table size: " << rightTable[0].size() << "\n";
		print(rightTable);
		*/

		// recursively assign the left child as a split where all values are < than the split
		root->leftChild = decisionTreeLearning(leftC, leftTable, leftIndices);
		// recursively assign the right child as a split where all values are >= than the split
		root->rightChild = decisionTreeLearning(rightC, rightTable, rightIndices);
	}
	return root;	// return the root back
}

// deletes the new data of the tree
void deleteTree ( Node* root ) 
{
	// if the subroot isn't null
    if(root!=nullptr)
    {
        deleteTree(root->leftChild);		// delete it's left
        deleteTree(root->rightChild);		// delete it's right
        delete(root);						// delete this
        if(root->leftChild!=nullptr)		
            root->leftChild=nullptr;		// set left null
        if(root->rightChild!=nullptr)		
            root->rightChild=nullptr;		// set right null
        root=nullptr;						// set this null
    }
} 

// opens the file from the string and puts the values into a 2D vector
void handleTestIn(vvd& v, const string in)
{
 	ifstream f;			// input filestream
 	f.open(in);			// open the training file

 	for (std::string line; std::getline(f, line); )
	{
    	std::istringstream iss(line);
    	v.emplace_back(std::istream_iterator<double>(iss),	// feed into the temp double from above
                   std::istream_iterator<double>());
	}
	f.close();					// close the file
}

// iterates through the ID3 tree and predicts what the classification is
int testLine(vd v, Node* root)
{
	// prediction as to what the classification is
	int predicton = 0;

	// iterate through the tree
	while (!root->isTerminal && (root->leftChild != nullptr || root->rightChild != nullptr)){
		
		//cout << "vec: " <<  v[root->splitColumn] << " split: " << root->split << "\n";
		
		// left child is less than the split
		if (v[root->splitColumn] < root->split)
		{
			root = root->leftChild;				// go left
			//cout << "left @ less than " << root->split << "\n";
		}

		// right child is greater than or equal to the split
		else if (v[root->splitColumn] >= root->split)
		{
			root = root->rightChild;			// go right
			//cout << "right @ greater than " << root->split << "\n";
		}

		if(root->isTerminal){				// if it's a terminal
			predicton = root->classifier;	// set the prediction
			//cout << "terminated @ " << predicton << "\n";
			break;							// break out of the loop
		}
	}
	return predicton;	// return the prediction
}

// returns the number of correctly predicted classifications
int testTree(Node* root, const vvd& v)
{	
	// the vector containing all the predictions of classifications 
	vi predictons;
	// the vector of their actual value
	vi actuals;
	// iterator for how many are correct
	int correct=0;
	// the column where our classifications in the vector are (4)
	int classColumn = v[0].size()-1;

	for (auto& r : v){

		//print(r);
		// iterate through and find the actual values
		actuals.push_back(r[classColumn]);
	
		// iterate through and find the actual values, put them in a vector
		predictons.push_back(testLine(r,root));
	}
	
	// print the vectors out 
	/*
	cout << "actuals\n";
	print(actuals);
	cout << "predictions\n";
	print(predictons);
	*/
	
	int arraySize = actuals.size();

	//cout << "size: " << arraySize << "\n";
	//print(actuals);
	//print(predictons);

	// figure out how many of them match
	for (int i=0; i<arraySize; i++){
		if (predictons[i] == actuals[i])
		{
			//cout << predictons[i] << " and " << actuals[i] << "\n";
			correct++;
		}
	}
	//exit(0);
	// return how many are correctly matched
	return correct;
}