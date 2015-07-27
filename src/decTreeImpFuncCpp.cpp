/*
An C++ source code that can be compiled by Rcpp and imported into R
Author: Li Qinglong 
Last update: 2015.03.17
William, your code is like a shit!!!
*/

/*******************************************************************************
********************************************************************************
Author: William Lai
Last update : 28.11.2005

Points to noted:
1. Implement as binary split 
2. Suit for binary, discrete, and continuous variables  
3. Rank the first three item    
4. Splitting  
   1. Considering: rank1 item
                   rank1, rank2 items
                   rank1, rank2, and rank3 items 
   2. Method: Top-k / Pairwise Entropy value
              Top-k / Pairwise Gini value
              Chi-square Test
              Statistical Test
              (-T [E:Entropy; G:Gini; C:Chi-square Test; S:Statistical Test)
   3. Stopping: no enhancement (pure node)
                if num of member < MIN_NODE_SIZE
5. Pruning
   1. Method: Gini Diversity Index
              Entropy value (Deviance)
              Statistical Test
6. Validation
   1. Method: CART cross-validation method is used to choose the best tree 
7. Performance Accessment based on validation data
   1. Method: Discrepancy measure based on pair comparisons 
              Foot Rule
              Misclassification Rate
              Area Under the ROC curve 
********************************************************************************
*******************************************************************************/

#include <iostream>
#include <iomanip>
#include <math.h>
#include <string>
#include <sstream>
#include <queue>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <algorithm>
#include <Rcpp.h>

#include "data.h"
#include "question.h"
#include "utility.h"
#include "input.h"
#include "myrand.h"

using namespace std;
using namespace Rcpp;
// Self defined Class: Question, Data, Utility, Input
// Self defined structs: pair_comp, node_info, tree_info, AUCpair

//**********************************************************************************
// Global variables
/* you can change the following parameters to fit the data */
const int MAX_ITEM_SIZE    = 10;//4;                  
const int MAX_RANK_SIZE    = 10;//2;  
int ITEM_SIZE = 4;
int RANK_SIZE = 2;
int CART_SIZE = 10; // number of cross validation in tree pruning
const double THRESHOLD = 0.0001;
const int SEED = 1400596410; // (unsigned)time(0); specify random seed for random number generation in data partition
const double MAX_VAL = 99999.9;//= minimum value for g-value in CART cross-validation

string DATA_FILE       = "input_01.txt"; // specify the input data file
string INFO_FILE       = "info_01.txt";      // specify the info of data 
// ofstream outfile("E:\\TDDOWNLOAD\\treeAUC_Cpp\\tree_output.txt"); // Li Qinglong, print the outputs in output.txt 
ofstream outfile; // Li Qinglong, print the outputs in output.txt 
/* you should re-compile the program once finished the change */
vector <double> alpha;  /* use for foot rule */
string mode       = "E"; //Spliting Criterion: E = Entropy G = Gini S = Statistical Test C = Chi-square test
string TMODE      = "L"; // When mode = "C" P = Pearson Chi-square test; L =  Likelihood ratio test
bool pairwise     = false;// use pairwise comparison model or not
bool PW_DATA      = false;// use pairwise data or top k-ranked data
string prediction = "M";// M : mean, F : frequent, C : center
int NODE_SIZE     = 573;//5000; 
int MIN_NODE_SIZE = 573;//5000; 
int RANK_ITEM     = 2;//2; top-q measure (1-3)
double ST_ALP    = 0.95; // level of significance of Statistical test
double CHI_ALP    = 0.2; // level of significance of Chi-square test
double TRAIN_PROP = 0.7;//= proportion of training data; effective only when CV_TEST = false
bool CV_TEST = false; // use 10-fold CV testing or not 
int TEST_STAGE = 0;  // the stage of the 10-fold CV testing; effective only when CV_TEST = true; value starts from 0 to V-1
Input qf_input;
Utility ut;
double yy[MAX_ITEM_SIZE][MAX_ITEM_SIZE]; // yy[i][j] = number of times which i beats j
double crank[MAX_ITEM_SIZE]; // central-ranked item for ranking [i]
// end of defining global variables
  //*****************************************************************************************************************

  // NEW METHOD for pairwise data for ENTROPY and GINI
  //calculate the pairwise data yyt[i][j], assume tied for non-ranked data
  // Pre: store the count of different rank of current node to array dataset
  void countData(double count[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE], vector <Data> node){
    for (int i=0 ; i<ITEM_SIZE ; i++)
      for (int j=0 ; j<ITEM_SIZE ; j++)
        for (int k=0 ; k<ITEM_SIZE ; k++)
          count[i][j][k] = 0.0;      
   
    for (size_t i=0 ; i<node.size() ; i++){
      int x1 = qf_input.find(node[i].getTarget(0));
      int x2 = qf_input.find(node[i].getTarget(1));
      int x3 = qf_input.find(node[i].getTarget(2));
      
      switch (node[i].getNOR()){
        case 1:           
          for (int a=0 ; a<ITEM_SIZE ; a++)
            for (int b=0 ; b<ITEM_SIZE ; b++)   
              if (a!=x1 && b!=x1 && a!=b)
                count[x1][a][b] += 1.0/((ITEM_SIZE-1)*(ITEM_SIZE-2));   
          break;   
        case 2:     
          for (int a=0 ; a<ITEM_SIZE ; a++)  
            if (a!=x1 && a!=x2) 
              count[x1][x2][a] += 1.0/(ITEM_SIZE-2); 
          break;
        default:
          count[x1][x2][x3] += 1;   
      } 
    }
  }


  // Pre: input count[i][j][k] from a dataset
  // Post: count data to give the pairwise frequency (yy[i][j] = item i > item j)
  void countpair(double yyy[MAX_ITEM_SIZE][MAX_ITEM_SIZE], double count[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE], int size){

    for (int i=0 ; i<ITEM_SIZE ; i++)
      for (int j=0 ; j<ITEM_SIZE ; j++)
        yyy[i][j] = 0.0;

    for (int i=0 ; i<ITEM_SIZE ; i++){
        for (int j=0; j<ITEM_SIZE; j++){
            for (int k=0; k<ITEM_SIZE; k++){
                for (int m=0; m<ITEM_SIZE; m++){
                    if (i != m){
                    yyy[i][m] += count[i][j][k];
                    }
                    if (j != m){
                           if (m != i){
                           yyy[j][m] += count[i][j][k];
                           }
                    }
                    if (k != m){
                           if (m != i){
                                 if (m != j){
                           yyy[k][m] += count[i][j][k];
                           }
                        }
                    }
                }
            }
        }
    }

    double temp_store;
    
    for (int i=0; i<ITEM_SIZE; i++){
        for (int j=0; j<ITEM_SIZE; j++){
            if (i != j){
                temp_store = size*1.0 - yyy[i][j] - yyy[j][i];
                yyy[i][j] += temp_store/2.0;
                yyy[j][i] += temp_store/2.0;
            }
        }
    }
  }
  /**** NEW METHOD, for both ENTROPY and GINI ****/

  // Pre: the first two rank is x1, x2
  // Post: return the impurity value by considering the item which ranked 3
  double value3(int x1, int x2, int n, double dataset[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE]){  
    double value = 0.0;
    if (mode == "E"){
      for (int i=0 ; i<ITEM_SIZE ; i++)       
        if (i!=x1 && i!=x2 && dataset[x1][x2][i]!=0) 
          value += (dataset[x1][x2][i]*1.0/n) * log(dataset[x1][x2][i]*1.0/n)/log(2.0);
    }  
    else { 
      for (int i=0 ; i<ITEM_SIZE ; i++) 
        if (i!=x1 && i!=x2 && dataset[x1][x2][i]!=0)
          value += (dataset[x1][x2][i]*1.0/n) * (dataset[x1][x2][i]*1.0/n); 
    }    
    return value;   
  }


  // Pre: the item which rank first is x1
  // Post: return the impurity value by considering the item which ranked 2 and 3
  double value2(int x1, int n, int y, double dataset[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE]){  
    double value = 0.0;  
    if (y == 2){     /* use rank1, rank2 item to rank */
      for (int i=0 ; i<ITEM_SIZE ; i++)
        if (i != x1){
          double tmp_count = 0.0;
          for (int j=0 ; j<ITEM_SIZE ; j++)
            tmp_count += dataset[x1][i][j];
          if (tmp_count != 0)
          {
            if (mode == "E")
              value += (tmp_count*1.0/n) * log(tmp_count*1.0/n)/log(2.0);
            else 
              value += (tmp_count*1.0/n) * (tmp_count*1.0/n);        
          }
        }    
    } 
    else {           /* use rank1, rank2, rank3 item to rank */
      for (int i=0 ; i<ITEM_SIZE ; i++)
        if (i != x1)
          value += value3(x1, i, n, dataset);  
    }
    return value;
  }


  // Pre: given the count of different combinations of a node
  // Post: return the impurity value of a node
  double getValue(int n, int y, double dataset[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE]){  
    double value = 0.0;  
    if(!PW_DATA){
        if (y == 1){     /* use rank1 item to rank */
          for (int i=0 ; i<ITEM_SIZE ; i++){
            double tmp_count = 0.0;
            for (int j=0 ; j<ITEM_SIZE ; j++)
              for (int k=0 ; k<ITEM_SIZE ; k++)
                tmp_count += dataset[i][j][k];
            if (tmp_count != 0)
            {
              if (mode == "E")
                value += (tmp_count*1.0/n) * log(tmp_count*1.0/n)/log(2.0);
              else 
                  value += (tmp_count*1.0/n) *(tmp_count*1.0/n);      
            }
          }    
        }
        else {           /* use more items to rank */
          for (int i=0 ; i<ITEM_SIZE ; i++)
            value += value2(i, n, y, dataset);
        }     
        return ((mode=="E")?-value:(1-value));
    }
    
    if(PW_DATA){ // pairwise data
     
      double yyy[MAX_ITEM_SIZE][MAX_ITEM_SIZE]; // transform dataset[][][] to pairwise data yyy[][]
      countpair(yyy, dataset, n);
    
    //finish transform
      for (int i=0; i<ITEM_SIZE; i++){
        for (int j=0; j<ITEM_SIZE; j++){
            if ((i != j) && (yyy[i][j] != 0)){
               if (mode == "E"){   
                  value += yyy[i][j]*1.0/n * log(yyy[i][j]*1.0/n)/log(2.0);
               }
               else{
                  value += (yyy[i][j]*1.0/n)*(yyy[i][j]*1.0/n);
               }
            }
        }
      }   
    
      value = (value*2.0/ITEM_SIZE*1.0)/(ITEM_SIZE*1.0-1.0);
      return ((mode=="E")?-value:(1.0-value));
    
    } // pairwise data end
    return 0.0;
  }
  /**** NEW METHOD END ****/  


  // Pre: input a tree generated by training data
  // Post: put all the testing data to the inputted tree
  void putTestData(vector <node_info>& nodeInfo, vector <Data> testData){
    for (size_t k=0 ; k<testData.size() ; k++){ 
      Data curr_data = testData[k];
      node_info curr_node = nodeInfo[0];
      (nodeInfo[0].smbr).push_back(curr_data);
   
      while (curr_node.fn != 1){  // do while curr_node is not a leaf
        string next_node_id;

        if (curr_node.st == 0){      // continuous split    
          if (curr_data.getInput(curr_node.sv0) < curr_node.sv1) 
            next_node_id = curr_node.id + "1";
          else next_node_id = curr_node.id + "2";  
        }    
        else {                      // discrete split
          bool match = false;
          for (size_t i=0 ; !match&&i<curr_node.sv2.size() ; i++) 
          if (qf_input.findInput(curr_node.sv0, curr_node.sv2[i]) == curr_data.getInput(curr_node.sv0))
            match = true; 
          if (match)
            next_node_id = curr_node.id + "1";
          else next_node_id = curr_node.id + "2";       
        }
        for (size_t n=0 ; n<nodeInfo.size() ; n++)
          if (nodeInfo[n].id == next_node_id){
            curr_node = nodeInfo[n];
            (nodeInfo[n].smbr).push_back(curr_data);
          } 
      }
    } 
  }


  // Pre: Access method: Discrepancy measure based on pair comparisons
  // Post: return the calculated discrepancy
  double calculateDiscrepancy1(const vector <pair_comp>& pairs, int size){
    double discrepancy = 0.0;
    for (size_t i=0 ; i<pairs.size() ; i++){
      double pe = pairs[i].exp_p/(1.0*size);
      double po = pairs[i].obs_p/(1.0*size);
      double temp1 = 0.5/(1.0*size);
      double temp2 = 1.0 - temp1;
      if (pe!= 0.0 && pe!=1.0)
        discrepancy += ((po-pe)*(po-pe))/(pe*(1.0-pe));
      if (pe==0.0 && po!=0.0)
        discrepancy += ((po-temp1)*(po-temp1))/(temp1*(1.0-temp1));
      if (pe==1.0)
        discrepancy += ((po-temp2)*(po-temp2))/((1.0-temp1)*temp1);
    }
    return discrepancy;
  }


  // Pre: Access method: Foot Rule 
  // Post: return the calculated discrepancy
  double calculateDiscrepancy2(double dist_matrix[MAX_RANK_SIZE+1][MAX_RANK_SIZE+1], vector <double> alpha, int size){
    vector<double> v0;
    for (unsigned i=0 ; i<RANK_SIZE ; i++)
      v0.push_back(i+1);
    if (ITEM_SIZE != RANK_SIZE)   
      v0.push_back((ITEM_SIZE+RANK_SIZE+1)*1.0/2);

    double discrepancy = 0.0;
    for (unsigned i=0 ; i<v0.size() ; i++)
      for (unsigned j=0 ; j<v0.size() ; j++)
        discrepancy += alpha[i]*dist_matrix[i][j]*fabs(v0[i]-v0[j])/size;
    return discrepancy;
  }


  // Pre: Access Method: Area Under ROC curve using training data (x=0), using validation data (x=1)
  // Post: return the calculated AUC for each pairwised items
  List assessAUC (const vector <node_info>& nodeInfo, int x){
    List ROC_List(ITEM_SIZE * (ITEM_SIZE - 1) / 2); // Edited by Li Qinglong
    CharacterVector List_Names(ROC_List.size());
    vector <AUCpair> auc_pair;
    auc_pair.clear();
    int size = 0;
    vector <Data> dataset;
    for (size_t n=0; n < nodeInfo.size(); n++){  // store all pwprob of each node in auc_pair
      double count[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE];
      double yy[MAX_ITEM_SIZE][MAX_ITEM_SIZE];
      if (nodeInfo[n].fn == 1){ 
        if (x==0){ 
          size += nodeInfo[n].mbr.size();
          countData(count,nodeInfo[n].mbr);
          countpair(yy,count,nodeInfo[n].mbr.size());
          for (int j=0; j<ITEM_SIZE; j++)
            for (int k=j+1; k<ITEM_SIZE; k++)
              auc_pair.push_back(AUCpair(j,k,nodeInfo[n].id,nodeInfo[n].mbr.size(),yy[j][k]/(1.0*nodeInfo[n].mbr.size())));
        }
        else if (x==1){
          size += nodeInfo[n].smbr.size();  // size of whole validation dataset
          countData(count,nodeInfo[n].mbr);
          countpair(yy,count,nodeInfo[n].mbr.size());
          for (int j=0; j<ITEM_SIZE; j++)
            for (int k=j+1; k<ITEM_SIZE; k++)
              if (nodeInfo[n].smbr.size()>0)
                auc_pair.push_back(AUCpair(j,k,nodeInfo[n].id,nodeInfo[n].smbr.size(),yy[j][k]/(1.0*nodeInfo[n].mbr.size())));
        }
      }
    }

    double AUC_avg = 0.0;
    int n_pair = 0;
    vector <AUCpair> sortpair;
    vector <double> tmp_p1;
    vector <double> tmp_p2;
    outfile<<"----------------------------------------------------------------------------"<<endl;
    if (x==0)
      outfile<<"||      Access method: Area Under the ROC curve (training data)            ||"<<endl;
    else if (x==1)
      outfile<<"||      Access method: Area Under the ROC curve (validation data)          ||"<<endl;
    outfile<<"----------------------------------------------------------------------------"<<endl;
    int pair_id = -1;
    for (int j=0; j<ITEM_SIZE; j++){
      for (int k=j+1; k<ITEM_SIZE; k++){
        pair_id++;
        n_pair++;
        sortpair.clear();
        tmp_p1.clear();
        tmp_p2.clear();
        double AUC = 0.0;
        double sen = 0.0;
        double spe = 0.0;
        double pw_size =0.0;
        for (size_t n=0; n<auc_pair.size(); n++){    // sort pwprob for each item pair in ascending order
          if (auc_pair[n].item1==j && auc_pair[n].item2==k){
            tmp_p1.push_back(auc_pair[n].pwprob);
            pw_size += auc_pair[n].pwprob*auc_pair[n].nsize;  // total count of j > k in data for each item pair
          }
        }
        sort(tmp_p1.begin(),tmp_p1.end());
        for (size_t n=0; n<tmp_p1.size(); n++)
          tmp_p2.push_back(tmp_p1[tmp_p1.size()-1-n]);  // sort pwprob for each item pair in descending order
        for (size_t n=0; n<tmp_p2.size();){
          for (size_t i=0; i<auc_pair.size(); i++){
            if (auc_pair[i].item1==j && auc_pair[i].item2==k){
              if (auc_pair[i].pwprob == tmp_p2[n] && n<tmp_p2.size()){
                sortpair.push_back(auc_pair[i]);
                n++;
              }
            }
          }
        }
        
        outfile<<"|\t\t AUC of item pair: "<<qf_input.findTarget(j)<<" is preferred to "<<qf_input.findTarget(k)<<"          \t\t\t\t\t|"<<endl;
        outfile<<"|     node id     leaf size     pw_prob     Sensitivity     1-Specificity   |"<<endl;
        outfile<<"----------------------------------------------------------------------------|"<<endl;
        double sen_size = 0.0; 
        double spe_size = 0.0; 
        double sen_0 = 0.0;
        double spe_0 = 0.0;

        // 此处需要将sen, spe输出出来
        // Li Qinglong
        stringstream ss1, ss2;
        ss1 << qf_input.findTarget(j);
        ss2 << qf_input.findTarget(k);
        List_Names[pair_id] = ss1.str() + " vs " + ss2.str();
        NumericMatrix matSpeVsSen(sortpair.size() + 1, 2);
        for (size_t i=0; i<sortpair.size(); i++){
          outfile<<"|     "<<sortpair[i].id<<"     \t"<<sortpair[i].nsize<<"   \t\t"<<fixed<<sortpair[i].pwprob<<"\t";
          sen_size += sortpair[i].nsize*sortpair[i].pwprob;
          spe_size += sortpair[i].nsize*(1.0-sortpair[i].pwprob);
          sen = sen_size/pw_size;
          spe = spe_size/(size*1.0-pw_size);
          outfile<<"\t  "<<sen<<"     \t\t"<<spe<<" \t|"<<endl;        
          AUC += (sen+sen_0)*(spe-spe_0)/2.0;  // calculate AUC using area method 
          matSpeVsSen(i + 1, 0) = spe;
          matSpeVsSen(i + 1, 1) = sen;
          sen_0 = sen;
          spe_0 = spe;
        }
        outfile << "|    Area under ROC = "<< AUC << "                                               |" << endl;
        outfile <<"----------------------------------------------------------------------------|"<<endl;
        AUC_avg += AUC;   // calculate the averaged AUC of all item pairs
        ROC_List[pair_id] = matSpeVsSen;
      }
    }
    outfile<<"Averaged AUC of "<<n_pair<<" item pairs = "<<AUC_avg/(1.0*n_pair) << endl;
    outfile<<endl;
    ROC_List.attr("names") = List_Names;
    return ROC_List;
  }


  // Pre: input dataSet (a node) 
  // Post: return first_rank, mean_rank
  void accessNode1(const vector<Data> dataSet, double first_rank[MAX_ITEM_SIZE], double mean_rank[MAX_ITEM_SIZE]){
    // double weight_rank[ITEM_SIZE]; 
    // what is the use of weight_rank?
    bool skip = false;
    if (first_rank[0] == -9.9)
      skip = true;
    for (int i=0 ; i<ITEM_SIZE ; i++){
      // weight_rank[i] = 0.0;
      first_rank[i] = 0.0;
      mean_rank[i] = 0.0;
    }   
    double count[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE];
    countData(count, dataSet);  
    // calculate first rank
    if (!skip)
      for (int i=0 ; i<ITEM_SIZE ; i++)
        for (int j=0 ; j<ITEM_SIZE ; j++)
          for (int k=0 ; k<ITEM_SIZE ; k++)
            if (i!=j && i!=k && j!=k)
              first_rank[i] += count[i][j][k];    
    // calculate mean rank
    double yy[MAX_ITEM_SIZE][MAX_ITEM_SIZE];
    countpair(yy, count, dataSet.size());

        for (int i=0 ; i<ITEM_SIZE ; i++){
          double tmp_count = 0.0;
          for (int j=0 ; j<ITEM_SIZE ; j++)
            if (i!=j)
              tmp_count += yy[i][j];     
          mean_rank[i] = ITEM_SIZE*1.0 - (tmp_count/dataSet.size()*1.0);
        } 
  }


  // Pre: input mean_rank
  // Post: return p_rank (predicted rank)
  vector <int> accessNode2(const vector<Data> dataSet, double first_rank[MAX_ITEM_SIZE], double mean_rank[MAX_ITEM_SIZE]){
    vector <int> p_rank;
    // unused variable tmp
    // double tmp = ITEM_SIZE*2.0;
    bool equal_mean = false;
    double count[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE];
    countData(count, dataSet);  
    
    if (prediction=="M"){
      vector <double> sort_rank(ITEM_SIZE);
      double tmp_mean[MAX_ITEM_SIZE];
      for (int i=0 ; i<ITEM_SIZE-1; i++){
        if (mean_rank[i] == mean_rank[i+1])
          equal_mean = true;
      }
      if (!equal_mean){
        for (int i=0 ; i<ITEM_SIZE; i++)
          tmp_mean[i] = mean_rank[i];
        for (int i=0 ; i<ITEM_SIZE; i++){
          p_rank.push_back(0);
          sort_rank[i] = tmp_mean[i];
        }
        sort(sort_rank.begin(), sort_rank.end()); // Sort the vector sort_rank according to the mean rank

        for (int i=0 ; i<ITEM_SIZE; i++){
          for (int j=0 ; j<ITEM_SIZE; j++){
            if (sort_rank[i] == tmp_mean[j]){
              p_rank[i] = j;
              tmp_mean[j]=-1.0;
              break;
            } 
          }
        }
      }
      if (equal_mean){
        p_rank.clear();
        double tmp_first[MAX_ITEM_SIZE];
        for (int i=0 ; i<ITEM_SIZE; i++)
          tmp_first[i] = first_rank[i];
        for (int i=0 ; i<ITEM_SIZE; i++){
          p_rank.push_back(0);
          sort_rank[i] = tmp_first[i];
        }
        sort(sort_rank.begin(), sort_rank.end()); // Sort the vector sort_rank according to the first rank
        for (int i=0 ; i<ITEM_SIZE; i++)
          for (int j=0 ; j<ITEM_SIZE; j++)
            if (sort_rank[i] == tmp_first[j]){
              p_rank[ITEM_SIZE-1-i] = j;
              tmp_first[j]=-1.0;
              break;
            }
      }
    } // end "M"
   
    if (prediction=="C"){
    // unused variable
    // int tmp_rank[3]; 
    double temp_dist[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE];
       for (int i=0 ; i<ITEM_SIZE ; i++){
           for (int j=0 ; j<ITEM_SIZE ; j++){
           if (j != i){
               for (int k=0 ; k<ITEM_SIZE ; k++){
               if (k != i && k != j){
                  for (int x=0 ; x<ITEM_SIZE ; x++){
                       for (int y=0 ; y<ITEM_SIZE ; y++){
                       if (y != x){
                           for (int z=0 ; z<ITEM_SIZE ; z++){
                           if (z != x && z != y){
                                 temp_dist[i][j][k] += count[x][y][z] * (((i==x)?0:1) + ((j==y)?0:1) + ((k==z)?0:1));
                           }
                           }
                       }
                       }
                   }             
               }
               }
           }
           }
       }
       p_rank.push_back(0);
       p_rank[0] = 0;
       p_rank.push_back(0);
       p_rank[1] = 1;
       p_rank.push_back(0);
       p_rank[2] = 2;     
       for (int i=0 ; i<ITEM_SIZE ; i++){
           for (int j=0 ; j<ITEM_SIZE ; j++){
           if (j != i){
               for (int k=0 ; k<ITEM_SIZE ; k++){
               if (k != i and k != j){    
                  if (temp_dist[i][j][k] < temp_dist[p_rank[0]][p_rank[1]][p_rank[2]]){
                  p_rank.push_back(0);
                  p_rank[0] = i;
                  p_rank.push_back(0);
                  p_rank[1] = j;
                  p_rank.push_back(0);
                  p_rank[2] = k;
                  }
               }
               }
           }
           }
       }                
    } // end "C"  
   
    if (prediction=="F"){
      double count_tmp2 = 0.0;
      switch (RANK_SIZE){
        case 1:
         p_rank.push_back(0);
         p_rank[0] = 0;
          for (int i=0 ; i<ITEM_SIZE ; i++){
            double count_tmp = 0.0;
            for (int j=0 ; j<ITEM_SIZE ; j++)
              for (int k=0 ; k<ITEM_SIZE ; k++)
                if (i!=j && j!=k && i!=k)
                  count_tmp += count[i][j][k];
            if (count_tmp > count_tmp2){
              count_tmp2 = count_tmp;
              p_rank.clear();
              p_rank.push_back(0);
              p_rank[0] = i;
            }
          }  
        break;
      case 2:
        for (int i=0 ; i<ITEM_SIZE ; i++)        
          for (int j=0 ; j<ITEM_SIZE ; j++){
            double count_tmp = 0.0;
            for (int k=0 ; k<ITEM_SIZE ; k++)
              if (i!=j && j!=k && i!=k)
                count_tmp += count[i][j][k];
            if (i!=j && count_tmp > count_tmp2){
              count_tmp2 = count_tmp;
              p_rank.clear();
              p_rank.push_back(0);
              p_rank[0] = i;
              p_rank.push_back(0);
              p_rank[1] = j;
            }
          }  
        break;         
      case 3:          
        for (int i=0 ; i<ITEM_SIZE ; i++)
          for (int j=0 ; j<ITEM_SIZE ; j++)
            for (int k=0 ; k<ITEM_SIZE ; k++)
              if (i!=j && j!=k && i!=k && count[i][j][k]>count_tmp2){
                count_tmp2 = count[i][j][k];
                p_rank.clear();
                p_rank.push_back(0);
                p_rank[0] = i;
                p_rank.push_back(0);
                p_rank[1] = j;
                p_rank.push_back(0);
                p_rank[2] = k;
              }
        break;        
    }   
    //     for (int i=0 ; i<ITEM_SIZE ; i++){
    //         for (int j=0 ; j<ITEM_SIZE ; j++){
    //         if (j != i){
    //             for (int k=0 ; k<ITEM_SIZE ; k++){
    //             if (k != i && k != j){ 
    //                if (count[i][j][k] > count[p_rank[0]][p_rank[1]][p_rank[2]]){
    //                    p_rank.clear();
    //                    p_rank.push_back(0);
    //                    p_rank[0] = i;
    //                    p_rank.push_back(0);
    //                    p_rank[1] = j;
    //                    p_rank.push_back(0);
    //                    p_rank[2] = k;
    //                    outfile << i << j << k << " " << count[i][j][k] << endl;
    //                }                   
    //                else if (count[i][j][k] = count[p_rank[0]][p_rank[1]][p_rank[2]]){
    //                  if (first_rank[i] > first_rank[p_rank[0]]){
    //                    p_rank.clear();
    //                    p_rank.push_back(0);
    //                    p_rank[0] = i;
    //                    p_rank.push_back(0);
    //                    p_rank[1] = j;
    //                    p_rank.push_back(0);
    //                    p_rank[2] = k;
    //                    outfile << i << j << k << " " << count[i][j][k] << endl;
    //                  }
    //                }    
    //             }
    //             }
    //         }
    //         }
    //     }
   }  // end "F"

    for (unsigned i=0 ; i<p_rank.size(); i++)  //for (int i=0 ; i<ITEM_SIZE ; i++)
      p_rank[i] = qf_input.findTarget(p_rank[i]);
    return p_rank;
  }


  // Pre: input count, p_rank
  // Post: update exp_p and obs_p for pairwise discrepancy measure, dist_matrix for foot rule
  void accessNode3(double count[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE], double dist_matrix[MAX_RANK_SIZE][MAX_RANK_SIZE], 
    vector <pair_comp>& pairs, vector <int> p_rank, double first_rank[MAX_ITEM_SIZE],
                   double mean_rank[MAX_ITEM_SIZE], int size){
      // fill dist_matrix
    double tmp[MAX_ITEM_SIZE]; 
    for (int i=0 ; i<RANK_SIZE ; i++){
      for (int j=0 ; j<ITEM_SIZE ; j++)
        tmp[j] = 0.0;
      for (int j=0 ; j<ITEM_SIZE ; j++)
        for (int k=0 ; k<ITEM_SIZE ; k++){
          switch (RANK_SIZE){
            case 1:
              if ((mean_rank[qf_input.find(p_rank[i])] != mean_rank[qf_input.find(p_rank[i+1])])
              || (first_rank[qf_input.find(p_rank[i])] != first_rank[qf_input.find(p_rank[i+1])]))
                tmp[0] += count[qf_input.find(p_rank[i])][j][k];
              else{
                int ntie = 0;
                for (int l=0 ; l<ITEM_SIZE ; l++)
                  if (mean_rank[qf_input.find(p_rank[l])] == mean_rank[qf_input.find(p_rank[l+1])]){
                    ntie += 1;
                    break;
                  }
                for (int l=0 ; l<=ntie ; l++)
                  tmp[0] += count[qf_input.find(p_rank[l])][j][k]/(1.0*ntie);
              }
              break;
            case 2:
              tmp[0] += count[qf_input.find(p_rank[i])][j][k];
              tmp[1] += count[j][qf_input.find(p_rank[i])][k]; 
              break;
            case 3:
              tmp[0] += count[qf_input.find(p_rank[i])][j][k];
              tmp[1] += count[j][qf_input.find(p_rank[i])][k];
              tmp[2] += count[j][k][qf_input.find(p_rank[i])];
              break;
          }
        }
      for (int j=0 ; j<RANK_SIZE ; j++)
        dist_matrix[i][j] += tmp[j];
    }  

    if (pairs.size() == 0)
      return;

    // fill comparison pair
    // expected prob.
    for (int i=0 ; i<ITEM_SIZE ; i++)
      for (int j=i+1 ; j<ITEM_SIZE ; j++)
        for (size_t k=0 ; k<pairs.size() ; k++)
          if (pairs[k].item1==qf_input.find(p_rank[i]) && pairs[k].item2==qf_input.find(p_rank[j])){
            if ((mean_rank[pairs[k].item1] != mean_rank[pairs[k].item2]) || 
                 first_rank[pairs[k].item1] != first_rank[pairs[k].item2])
              pairs[k].exp_p += 1.0*size;
            else if ((mean_rank[pairs[k].item1] == mean_rank[pairs[k].item2]) && 
                      first_rank[pairs[k].item1] == first_rank[pairs[k].item2])
              pairs[k].exp_p += 1.0*size/2.0;
          }
    // observed prob.
    for (int i=0 ; i<ITEM_SIZE ; i++)
      for (int j=i+1 ; j<ITEM_SIZE ; j++)
        for (size_t k=0 ; k<pairs.size() ; k++)
          if (pairs[k].item1==i && pairs[k].item2==j){
            for (int l=0 ; l<ITEM_SIZE ; l++)
              if (l!=i && l!=j)
                pairs[k].obs_p += (count[i][j][l]+count[i][l][j]+count[l][i][j]);

            for (int l=0 ; l<ITEM_SIZE ; l++)
              for (int m=0 ; m<ITEM_SIZE ; m++)
                if (l!=i && l!=j && m!=i && m!=j && l!=m)
                  pairs[k].obs_p += (count[i][l][m]+count[l][i][m]+count[l][m][i]);
          }
  }


  // Accessment by Gini Diversity Index
  // Post : return = count_of_test_data * (1 + sum_of_p^2_of_train_data + 2*train_data)
  double accessNode6(const node_info& nodeInfo){  
    double cnt1[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE];     
    double cnt2[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE];  
    double r = 0.0;
    double s = 0.0;  
    countData(cnt1, nodeInfo.mbr); 
    countData(cnt2, nodeInfo.smbr); 
    //if not pairwise impurity measure
    if (!PW_DATA){
      if (mode == "E"){
        switch (RANK_ITEM){
          case 1:   
            for (int i=0 ; i<ITEM_SIZE ; i++){
              double r11 = 0.0;
              double r12 = 0.0;
              for (int j=0 ; j<ITEM_SIZE ; j++)
                for (int k=0 ; k<ITEM_SIZE ; k++) 
                  if (i!=j && j!=k && i!=k){
                    r11 += cnt1[i][j][k];
                    r12 += cnt2[i][j][k];
                  }  
              if (r11 > 0.0)
                r -= r12 * log(r11/(nodeInfo.mbr).size())/log(2.0);  
            }          
            break;
          case 2: 
            for (int i=0 ; i<ITEM_SIZE ; i++)
              for (int j=0 ; j<ITEM_SIZE ; j++){
                double r21 = 0.0;
                double r22 = 0.0;
                for (int k=0 ; k<ITEM_SIZE ; k++) 
                  if (i!=j && j!=k && i!=k){
                    r21 += cnt1[i][j][k];
                    r22 += cnt2[i][j][k];
                  }  
                if (r21 > 0.0)  
                  r -= r22 * log(r21/(nodeInfo.mbr).size())/log(2.0);    
              }                   
            break;
          case 3:  
            for (int i=0 ; i<ITEM_SIZE ; i++)
              for (int j=0 ; j<ITEM_SIZE ; j++)
                for (int k=0 ; k<ITEM_SIZE ; k++) 
                  if (i!=j && j!=k && i!=k && cnt1[i][j][k]>0.0)
                    r -= cnt2[i][j][k] * log(cnt1[i][j][k]/(nodeInfo.mbr).size())/log(2.0);     
            break; 
        }           
      }
      else if (mode == "G"){
        switch (RANK_ITEM){
          case 1:
            for (int i=0 ; i<ITEM_SIZE ; i++){
              double s1 = 0.0;
              for (int j=0 ; j<ITEM_SIZE ; j++)
                for (int k=0 ; k<ITEM_SIZE ; k++)
                  if (i!=j && j!=k && i!=k)
                    s1 += cnt1[i][j][k];
              s += s1*s1;
            }  
            s /= ((nodeInfo.mbr).size()*(nodeInfo.mbr).size()); 

            for (int i=0 ; i<ITEM_SIZE ; i++){
              double r11 = 0.0;
              double r12 = 0.0;
              for (int j=0 ; j<ITEM_SIZE ; j++)
                for (int k=0 ; k<ITEM_SIZE ; k++) 
                  if (i!=j && j!=k && i!=k){
                    r11 += cnt1[i][j][k];
                    r12 += cnt2[i][j][k];
                  }  
              r += r12*(1+s-2.0*r11/(nodeInfo.mbr).size());     
            }   
            break;
          case 2:
            for (int i=0 ; i<ITEM_SIZE ; i++)
              for (int j=0 ; j<ITEM_SIZE ; j++){
                double s2 = 0.0;
                for (int k=0 ; k<ITEM_SIZE ; k++)
                  if (i!=j && j!=k && i!=k)
                    s2 += cnt1[i][j][k];
                s += s2*s2;
              }  
            s /= ((nodeInfo.mbr).size()*(nodeInfo.mbr).size()); 
       
            for (int i=0 ; i<ITEM_SIZE ; i++)
              for (int j=0 ; j<ITEM_SIZE ; j++){
                double r21 = 0.0;
                double r22 = 0.0;
                for (int k=0 ; k<ITEM_SIZE ; k++) 
                  if (i!=j && j!=k && i!=k){
                    r21 += cnt1[i][j][k];
                    r22 += cnt2[i][j][k];
                  }  
                r += r22*(1+s-2.0*r21/(nodeInfo.mbr).size());     
              }       
            break;
          case 3:   
            for (int i=0 ; i<ITEM_SIZE ; i++)
              for (int j=0 ; j<ITEM_SIZE ; j++)
                for (int k=0 ; k<ITEM_SIZE ; k++)
                  if (i!=j && j!=k && i!=k)
                    s += cnt1[i][j][k]*cnt1[i][j][k];
            s /= ((nodeInfo.mbr).size()*(nodeInfo.mbr).size()); 
    
            for (int i=0 ; i<ITEM_SIZE ; i++)
              for (int j=0 ; j<ITEM_SIZE ; j++)
                for (int k=0 ; k<ITEM_SIZE ; k++) 
                  if (i!=j && j!=k && i!=k)
                    r += cnt2[i][j][k]*(1+s-2.0*cnt1[i][j][k]/(nodeInfo.mbr).size());  
          break;
        }
      }     
    }  // if not pairwise impurity measure end

     //if pairwise impurity measure 
    if (PW_DATA){
      double yy1[MAX_ITEM_SIZE][MAX_ITEM_SIZE];  
      double yy2[MAX_ITEM_SIZE][MAX_ITEM_SIZE];
      double ss[MAX_ITEM_SIZE][MAX_ITEM_SIZE];
      r = 0.0;
      for (int i=0; i<ITEM_SIZE; i++){
        for (int j=0; j<ITEM_SIZE; j++){
            ss[i][j] = 0.0;
        }
      }
      countpair(yy1,cnt1,(nodeInfo.mbr).size());
      countpair(yy2,cnt2,(nodeInfo.smbr).size());

      if (mode == "G"){
        for (int i=0; i<ITEM_SIZE; i++){
          for (int j=0; j<ITEM_SIZE; j++){
            if (i != j){
              ss[i][j] = (yy1[i][j]/(nodeInfo.mbr).size())*(yy1[i][j]/(nodeInfo.mbr).size()) + 
                         (yy1[j][i]/(nodeInfo.mbr).size())*(yy1[j][i]/(nodeInfo.mbr).size());
            }
          }
        }
      }
      for (int i=0; i<ITEM_SIZE; i++){
        for (int j=0; j<ITEM_SIZE; j++){
          if (i != j){
            if (mode == "E"){
              if (yy1[i][j] > 0.0)
                r -= yy2[i][j] * log(yy1[i][j]/(nodeInfo.mbr).size())/log(2.0);
            }
            else if (mode == "G"){
              r += yy2[i][j] * (1.0 + ss[i][j] - 2.0*yy1[i][j]/(nodeInfo.mbr).size());
            }                
          }
        }
      }   
    }  // if pairwise impurity measure end

    //   r = 0.0; // testing testing
    //  if (mode == "S" || mode == "C" || mode == "G" || mode == "E"){  //testing testing
    if (mode == "S" || mode == "C"){ 
      double first_rank[MAX_ITEM_SIZE];
      double mean_rank[MAX_ITEM_SIZE];
      double dist_matrix[MAX_RANK_SIZE][MAX_RANK_SIZE];    
      int size = nodeInfo.smbr.size();
      vector <pair_comp> dummyPairs;
      first_rank[0] = -9.9;
      for (int i=0 ; i<RANK_SIZE ; i++)
        for (int j=0 ; j<RANK_SIZE ; j++) 
          dist_matrix[i][j] = 0.0;
      accessNode1(nodeInfo.mbr, first_rank, mean_rank);
      vector<int> p_rank = accessNode2(nodeInfo.mbr, first_rank, mean_rank);
      accessNode3(cnt2, dist_matrix, dummyPairs, p_rank, first_rank, mean_rank, size);  

      double dist_matrix2[MAX_RANK_SIZE+1][MAX_RANK_SIZE+1]; 
      for (int i=0 ; i<RANK_SIZE ; i++)
        for (int j=0 ; j<RANK_SIZE ; j++) 
          dist_matrix2[i][j] = dist_matrix[i][j];            
      double cnt2 = 0.0;
      for (int i=0 ; i<RANK_SIZE ; i++){
        double cnt1 = 0.0;
        for (int j=0 ; j<RANK_SIZE ; j++)  
          cnt1 += dist_matrix[i][j];
        cnt2 += (size*1.0 - cnt1); 
        dist_matrix2[i][RANK_SIZE] = (size*1.0 - cnt1);
      }
      for (int i=0 ; i<RANK_SIZE ; i++){ 
        double cnt3 = 0.0; 
        for (int j=0 ; j<RANK_SIZE ; j++)  
          cnt3 += dist_matrix[j][i];    
        dist_matrix2[RANK_SIZE][i] = (size*1.0 - cnt3);
      }    
      if (RANK_SIZE != ITEM_SIZE)
        dist_matrix2[RANK_SIZE][RANK_SIZE] = ((ITEM_SIZE-RANK_SIZE)*size*1.0 - cnt2); 
      else 
        dist_matrix2[RANK_SIZE][RANK_SIZE] = 0; 
         
      r = calculateDiscrepancy2(dist_matrix2, alpha, 1);
    }
    return r;     
  } 


  // Pre: input left node & right node
  // Post: return the new value (by Gini/Entropy) 
  double newValue(const vector<Data>& dataSet1, const vector<Data>& dataSet2, const vector <vector <double> >& invC){
    double new_value = 0.0;
    if (mode == "E" || mode == "G"){  
      double count[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE]; 
      countData(count, dataSet1);    /* Leave #1 */
      new_value += getValue(dataSet1.size(), RANK_ITEM, count) * dataSet1.size() / (dataSet1.size()+dataSet2.size());  
      countData(count, dataSet2);    /* Leave #2 */
      new_value += getValue(dataSet2.size(), RANK_ITEM, count) * dataSet2.size() / (dataSet1.size()+dataSet2.size()); 
    }
    else if (mode == "S"){
      double first_rank[MAX_ITEM_SIZE];
      double mean_rank1[MAX_ITEM_SIZE];
      double mean_rank2[MAX_ITEM_SIZE];
      first_rank[0] = -9.9;
      accessNode1(dataSet1, first_rank, mean_rank1); /* Leave #1 */    
      accessNode1(dataSet2, first_rank, mean_rank2); /* Leave #2 */  
      vector <double> rank_diff;
      for (int i=0 ; i<ITEM_SIZE-1 ; i++)
        rank_diff.push_back(mean_rank1[i]-mean_rank2[i]);  
      vector <double> vc = ut.multColMatrix(invC, rank_diff);
      new_value = ut.multRowCol(rank_diff, vc);
      new_value *= (dataSet1.size()*dataSet2.size()*1.0)/(dataSet1.size()+dataSet2.size());
      new_value *= -1.0;  /* return the -ve value */
    }
    else if (mode == "C"){
      double count1[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE]; 
      double count2[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE]; 
      double col_sum[2];
      countData(count1, dataSet1);    /* Leave #1 */
      countData(count2, dataSet2);    /* Leave #2 */
      if (!PW_DATA){
        double row_sum1[MAX_ITEM_SIZE];
        double row_sum2[MAX_ITEM_SIZE][MAX_ITEM_SIZE];
        double row_sum3[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE];
        double data_size;            
        for (int i=0 ; i<ITEM_SIZE ; i++){
          row_sum1[i] = 0.0;
          for (int j=0 ; j<ITEM_SIZE ; j++){
            row_sum2[i][j] = 0.0;
            for (int k=0 ; k<ITEM_SIZE ; k++)
              row_sum3[i][j][k]=0.0;
          }
        }
        col_sum[0]=0.0;
        col_sum[1]=0.0;
        for (int i=0 ; i<ITEM_SIZE ; i++){
          for (int j=0 ; j<ITEM_SIZE ; j++){
            for (int k=0 ; k<ITEM_SIZE ; k++){
              col_sum[0]+=count1[i][j][k];
              col_sum[1]+=count2[i][j][k];
            }
          }
        }               
        if ((col_sum[0]==0.0) || (col_sum[1]==0.0)){
          new_value = 0.0;
          return new_value;
        }
        data_size=col_sum[0]+col_sum[1];
    // outfile << col_sum[0] << " " << col_sum[1] << " " << data_size << " " <<endl;
        switch (RANK_ITEM){
        case 1:
          for (int i=0 ; i<ITEM_SIZE ; i++){
            for (int j=0 ; j<ITEM_SIZE ; j++){
              for (int k=0 ; k<ITEM_SIZE ; k++){
                if(i!=j && i!=k && j!=k)
                  row_sum1[i]+=count1[i][j][k]+count2[i][j][k];
              }
            }
    // outfile << "item "<<i+1<<" "<<row_sum1[i]<< endl;
          }    
          for (int i=0 ; i<ITEM_SIZE ; i++){
            double chi11 = 0.0;
            double chi12 = 0.0;
            for (int j=0 ; j<ITEM_SIZE ; j++)
              for (int k=0 ; k<ITEM_SIZE ; k++)
                if(i!=j && i!=k && j!=k){
                  chi11 += count1[i][j][k];
                  chi12 += count2[i][j][k];
                }
            if (row_sum1[i] > 0.0){
              if (TMODE == "P"){
                new_value+=(chi11-row_sum1[i]*col_sum[0]/data_size)*(chi11-row_sum1[i]*col_sum[0]/data_size)
                            /(row_sum1[i]*col_sum[0]/data_size);
                new_value+=(chi12-row_sum1[i]*col_sum[1]/data_size)*(chi12-row_sum1[i]*col_sum[1]/data_size)
                            /(row_sum1[i]*col_sum[1]/data_size);
              }
              if (TMODE == "L"){
                if (chi11 > 0.0)
                  new_value+=2.0*(chi11)*log(chi11/(row_sum1[i]*col_sum[0]/data_size));
                if (chi12 > 0.0)
                  new_value+=2.0*(chi12)*log(chi12/(row_sum1[i]*col_sum[1]/data_size));
              }
            }
          }  
        break;
        case 2: 
          for (int i=0 ; i<ITEM_SIZE ; i++){
            for (int j=0 ; j<ITEM_SIZE ; j++){
              for (int k=0 ; k<ITEM_SIZE ; k++){
                if(i!=j && i!=k && j!=k)
                  row_sum2[i][j]+=count1[i][j][k]+count2[i][j][k];
              }
    //  outfile << "item "<<i+1<<" "<<"item "<<j+1<<" "<<row_sum2[i][j]<< endl;
            }
          }    
          for (int i=0 ; i<ITEM_SIZE ; i++)
            for (int j=0 ; j<ITEM_SIZE ; j++){
              double chi21 = 0.0;
              double chi22 = 0.0;
              for (int k=0 ; k<ITEM_SIZE ; k++)
                if(i!=j && i!=k && j!=k){
                  chi21 += count1[i][j][k];
                  chi22 += count2[i][j][k];
                }
              if (row_sum2[i][j] > 0.0){
                if (TMODE == "P"){
                  new_value+=(chi21-row_sum2[i][j]*col_sum[0]/data_size)*(chi21-row_sum2[i][j]*col_sum[0]/data_size)
                             /(row_sum2[i][j]*col_sum[0]/data_size);
                  new_value+=(chi22-row_sum2[i][j]*col_sum[1]/data_size)*(chi22-row_sum2[i][j]*col_sum[1]/data_size)
                             /(row_sum2[i][j]*col_sum[1]/data_size);
                }
                if (TMODE == "L"){
                  if (chi21 > 0.0)
                    new_value+=2.0*(chi21)*log(chi21/(row_sum2[i][j]*col_sum[0]/data_size));
                  if (chi22 > 0.0)
                    new_value+=2.0*(chi22)*log(chi22/(row_sum2[i][j]*col_sum[1]/data_size));
                }
              }
            }  
        break;
        case 3: 
          for (int i=0 ; i<ITEM_SIZE ; i++){
            for (int j=0 ; j<ITEM_SIZE ; j++){
              for (int k=0 ; k<ITEM_SIZE ; k++){
                if(i!=j && i!=k && j!=k){
                  row_sum3[i][j][k]+=count1[i][j][k]+count2[i][j][k];
    //  outfile << row_sum3[i][j][k] << endl;
                }
              }
            }
          }     
          for (int i=0 ; i<ITEM_SIZE ; i++){
            for (int j=0 ; j<ITEM_SIZE ; j++){
              for (int k=0 ; k<ITEM_SIZE ; k++){
                if(i!=j && i!=k && j!=k){
                  if (row_sum3[i][j][k] > 0.0){
                    if (TMODE == "P"){
                      new_value+=(count1[i][j][k]-row_sum3[i][j][k]*col_sum[0]/data_size)*
                      (count1[i][j][k]-row_sum3[i][j][k]*col_sum[0]/data_size)/(row_sum3[i][j][k]*col_sum[0]/data_size);
                      new_value+=(count2[i][j][k]-row_sum3[i][j][k]*col_sum[1]/data_size)*
                      (count2[i][j][k]-row_sum3[i][j][k]*col_sum[1]/data_size)/(row_sum3[i][j][k]*col_sum[1]/data_size);
                    }
                    if (TMODE == "L"){
                      if (count1[i][j][k] > 0.0)
                        new_value+=2.0*(count1[i][j][k])*log(count1[i][j][k]/(row_sum3[i][j][k]*col_sum[0]/data_size));
                      if (count2[i][j][k] > 0.0)
                        new_value+=2.0*(count2[i][j][k])*log(count2[i][j][k]/(row_sum3[i][j][k]*col_sum[1]/data_size));
                    }
                  }
                }
              }
            }
          }                                                                      
          break;
        }
      } // end if not pw_data
      if (PW_DATA){
        double yy1[MAX_ITEM_SIZE][MAX_ITEM_SIZE]; 
        double yy2[MAX_ITEM_SIZE][MAX_ITEM_SIZE];
        double row_sum[MAX_ITEM_SIZE][MAX_ITEM_SIZE];
        double data_size;
        for (int i=0 ; i<ITEM_SIZE ; i++)
          for (int j=0 ; j<ITEM_SIZE ; j++)
            row_sum[i][j]=0;

        col_sum[0]=0;
        col_sum[1]=0;
        countpair(yy1, count1, dataSet1.size());
        countpair(yy2, count2, dataSet2.size());
        for (int i=0 ; i<ITEM_SIZE ; i++){
          for (int j=0 ; j<ITEM_SIZE ; j++){
            if (i != j){
    // outfile << setprecision(5)<< yy1[i][j] << " " << yy2[i][j] << " " <<endl;
              col_sum[0]+=yy1[i][j];
              col_sum[1]+=yy2[i][j];
            }
          }
        }
        data_size=col_sum[0]+col_sum[1];
    // outfile << col_sum[0] << " " << col_sum[1] << " " << data_size << " " <<endl;
        for (int i=0 ; i<ITEM_SIZE ; i++){
          for (int j=0 ; j<ITEM_SIZE ; j++){
            if (i != j){
              row_sum[i][j]= yy1[i][j]+yy2[i][j];                          
            }
          }
        }
              
        for (int i=0 ; i<ITEM_SIZE ; i++){
          for (int j=0 ; j<ITEM_SIZE ; j++){
            if ((i != j) && (row_sum[i][j]>0.0)){
              if (TMODE == "P"){
                new_value += (yy1[i][j]-row_sum[i][j]*col_sum[0]/data_size)*(yy1[i][j]-
                              row_sum[i][j]*col_sum[0]/data_size)/(row_sum[i][j]*col_sum[0]/data_size);
                new_value += (yy2[i][j]-row_sum[i][j]*col_sum[1]/data_size)*(yy2[i][j]-
                              row_sum[i][j]*col_sum[1]/data_size)/(row_sum[i][j]*col_sum[1]/data_size);
              }
              if (TMODE == "L"){
                if (yy1[i][j] > 0.0)
                  new_value+=2.0*(yy1[i][j])*log(yy1[i][j]/(row_sum[i][j]*col_sum[0]/data_size));
                if (yy2[i][j] > 0.0)
                  new_value+=2.0*(yy2[i][j])*log(yy2[i][j]/(row_sum[i][j]*col_sum[1]/data_size));
              }
            }
          }
        }
      }
      new_value *= -1.0;  /* return the -ve value */  
    }      
    return new_value;
  }


  // Pre: input datasets of the 2 child nodes arised from a split 
  // Post: return the degree of freedom of Chi-square statistics of that split
  int df(vector <Data> dataSet1, vector <Data> dataSet2){
    double count1[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE]; 
    double count2[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE]; 
    int deg = 0; // calculate degree of freedom of Chisq statistic
    countData(count1, dataSet1);    /* Leave #1 */
    countData(count2, dataSet2);    /* Leave #2 */
    if (!PW_DATA){
      double row_sum1[MAX_ITEM_SIZE];
      double row_sum2[MAX_ITEM_SIZE][MAX_ITEM_SIZE];
      double row_sum3[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE];
      for (int i=0 ; i<ITEM_SIZE ; i++){
        row_sum1[i] = 0.0;
        for (int j=0 ; j<ITEM_SIZE ; j++){
          row_sum2[i][j] = 0.0;
          for (int k=0 ; k<ITEM_SIZE ; k++)
            row_sum3[i][j][k]=0.0;
        }
      }
      switch (RANK_ITEM){
      case 1:
        for (int i=0 ; i<ITEM_SIZE ; i++){
          for (int j=0 ; j<ITEM_SIZE ; j++){
            for (int k=0 ; k<ITEM_SIZE ; k++){
              if(i!=j && i!=k && j!=k)
                row_sum1[i]+=count1[i][j][k]+count2[i][j][k];
            }
          }
        }    
        for (int i=0 ; i<ITEM_SIZE ; i++)
          if (row_sum1[i] > 0.0)
              deg += 1;
      break;
      case 2: 
        for (int i=0 ; i<ITEM_SIZE ; i++){
          for (int j=0 ; j<ITEM_SIZE ; j++){
            for (int k=0 ; k<ITEM_SIZE ; k++){
              if(i!=j && i!=k && j!=k)
                row_sum2[i][j]+=count1[i][j][k]+count2[i][j][k];
            }
          }
          for (int j=0 ; j<ITEM_SIZE ; j++){
            if (row_sum2[i][j] > 0.0)
              deg += 1;
          }
        }    
      break;
      case 3: 
        for (int i=0 ; i<ITEM_SIZE ; i++){
          for (int j=0 ; j<ITEM_SIZE ; j++){
            for (int k=0 ; k<ITEM_SIZE ; k++){
              if(i!=j && i!=k && j!=k){
                row_sum3[i][j][k]+=count1[i][j][k]+count2[i][j][k];
              }
              if (row_sum3[i][j][k] > 0.0)
                deg += 1;
            }
          }
        }      
      break;
      }
    }
    if (PW_DATA){
      for (int i=0 ; i<ITEM_SIZE ; i++){
        for (int j=0 ; j<ITEM_SIZE ; j++){
          if (i < j)
            deg += 1;
        }
      }
    }
    return deg-1;
  }
    
  // Pre: input a data set
  // Post: return the inverse of C
  vector <vector <double> > getMatInv(const vector<Data>& dataSet){  
    vector <double> v1;
    vector <vector <double> > v2, v3;
    for (int i=0 ; i<ITEM_SIZE-1 ; i++)
      v1.push_back(0.0);
    for (int i=0 ; i<ITEM_SIZE-1 ; i++)
      v2.push_back(v1);
    // (1/n) * sum_of ([r]*[r_t]) 
    for (size_t k=0 ; k<dataSet.size() ; k++){
      vector <double> vv1;
      for (int i=0 ; i<ITEM_SIZE-1 ; i++){   // only (K-1) items are used to formulate the matrix
        int rank = dataSet[k].getItemRank(qf_input.findTarget(i));
        if (rank == -1)
          vv1.push_back((ITEM_SIZE+RANK_ITEM+1)/2);
        else vv1.push_back(rank);
      }
      v2 = ut.vecToMatrix(vv1, v2);
    }                                            // v2 (M) = matrix of summation of [R]*[Rt]/size
    v2 = ut.multMatrix(-1.0/dataSet.size(), v2); // v2     = -M  
    // [R]*[R_t] 
    double first_rank[MAX_ITEM_SIZE];
    double mean_rank[MAX_ITEM_SIZE];
    first_rank[0] = -9.9;
    accessNode1(dataSet, first_rank, mean_rank);
    for (int i=0 ; i<ITEM_SIZE-1 ; i++)
      v1[i] = mean_rank[i];
    v2 = ut.vecToMatrix(v1, v2);                         // v2 = matrix of [R]*[R_t] - M
    v2 = ut.multMatrix(-1.0, v2);                        // v2 = C
    // LU matrix
    vector <vector <double> > ludmat = ut.luDecomp(v2);  // ludmat = LU matrix of v2
    v2 = ut.backSub(ludmat);                             // v2 = invC
    /*outfile<<"invC:"<<endl;
    for (size_t i1=0 ; i1<v2.size() ; i1++){
      for (size_t i2=0 ; i2<v2.size() ; i2++)
        outfile<<v2[i1][i2]<<",";
      outfile<<endl;
    }*/
    return v2;
  }


  // Pre: input first_rank, mean_rank, size, id of a node
  // Post: print the above infomation
  void printNode1(double first_rank[MAX_ITEM_SIZE], double mean_rank[MAX_ITEM_SIZE], int size, 
    const string id, vector<int> p_rank){
    outfile<<id<<"\t Size= "<<size<<"\t";
    if (p_rank.size() > 0){
      outfile<< "Predicted Order: ["<<p_rank[0];
      for (unsigned i=1 ; i<p_rank.size() ; i++) // for (int i=1 ; i<ITEM_SIZE ; i++)
        outfile<<","<<p_rank[i];
      outfile<<"]"<<endl;
    }  
    outfile<<"Rank1%: ";
    for (int i=0 ; i<ITEM_SIZE ; i++)
      outfile<<qf_input.findTarget(i)<<"("<<setprecision(3)<<first_rank[i]/size<<"),";  
    outfile<<"\t MeanRank: ";
    for (int i=0 ; i<ITEM_SIZE ; i++)
      outfile<<qf_input.findTarget(i)<<"("<<mean_rank[i]<<setprecision(3)<<"),";
    outfile<<endl;
  }


  // Pre: input the count, size of a node
  // Post: print different rankings frequency
  void printNode2(double count[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE], int size){
    outfile<<"RankFreq: "<<setprecision(3);
    switch (RANK_ITEM){
      case 1:
        for (int i=0 ; i<ITEM_SIZE ; i++){
          double count_tmp = 0.0;
          for (int j=0 ; j<ITEM_SIZE ; j++)
            for (int k=0 ; k<ITEM_SIZE ; k++)
              if (i!=j && j!=k && i!=k)
                count_tmp += count[i][j][k];
          outfile<<"{"<<qf_input.findTarget(i)<<"}("<<count_tmp<<","<<1.0*count_tmp/size<<"),";
        }  
        break;
      case 2:
        for (int i=0 ; i<ITEM_SIZE ; i++)        
          for (int j=0 ; j<ITEM_SIZE ; j++){
            double count_tmp = 0.0;
            for (int k=0 ; k<ITEM_SIZE ; k++)
              if (i!=j && j!=k && i!=k)
                count_tmp += count[i][j][k];
            if (i!=j)
              outfile<<"{"<<qf_input.findTarget(i)<<qf_input.findTarget(j)<<"}("<<count_tmp<<","<<1.0*count_tmp/size<<"),";
          }  
        break;         
      case 3:          
        for (int i=0 ; i<ITEM_SIZE ; i++)
          for (int j=0 ; j<ITEM_SIZE ; j++)
            for (int k=0 ; k<ITEM_SIZE ; k++)
              if (i!=j && j!=k && i!=k && count[i][j][k]>0)
                outfile<<"{"<<qf_input.findTarget(i)<<qf_input.findTarget(j)<<qf_input.findTarget(k)
                    <<"}("<<count[i][j][k]<<","<<1.0*count[i][j][k]/size<<setprecision(3)<<"),";
        break;        
    }            

    // count pairwise frequency by transforming dataset[][][] to pairwise data yyy[][]
      double yyy[MAX_ITEM_SIZE][MAX_ITEM_SIZE]; 
      countpair(yyy,count,size);

      //print the pairwise frequency count
      outfile<<endl<<"Pairwise Freq: "<<setprecision(3);
      for (int i=0 ; i<ITEM_SIZE ; i++)        
        for (int j=0 ; j<ITEM_SIZE ; j++){
          double count_tmp = 0.0;
            if (i!=j){
              count_tmp += yyy[i][j];
              outfile<<"{"<<qf_input.findTarget(i)<<qf_input.findTarget(j)<<"}("<<1.0*count_tmp/size<<"),";
            }
        }
      outfile<<endl<<" - -"<<endl;
  }


  // Post: Print the header of report
  void printHeader(int train_size, int val_size, double st_val){
    outfile<<"----------------------------------------------------------------------------"<<endl;
    if (mode=="E" && !PW_DATA)
      outfile<<"||      Spliting criterion: Entropy      \t\t\t\t  ||"<<endl;
    else if (mode=="G" && !PW_DATA) 
      outfile<<"||      Spliting criterion: Gini         \t\t\t\t  ||"<<endl;
    else if (mode=="E" && PW_DATA)
      outfile<<"||      Spliting criterion: Pairwise Entropy \t\t\t\t  ||"<<endl;
    else if (mode=="G" && PW_DATA) 
      outfile<<"||      Spliting criterion: Pairwise Gini    \t\t\t\t  ||"<<endl;
    else if (mode=="S") 
      outfile<<"||      Spliting criterion: Statistical Test\t\t\t\t  ||"<<endl;
    else if (mode=="C" && TMODE=="P" && !PW_DATA) 
      outfile<<"||      Spliting criterion: Chi-square Test\t\t\t\t  ||"<<endl;    
    else if (mode=="C" && TMODE=="P" && PW_DATA) 
      outfile<<"||      Spliting criterion: Pairwise Chisq Test\t\t\t\t  ||"<<endl;    
    else if (mode=="C" && TMODE=="L" && !PW_DATA) 
      outfile<<"||      Spliting criterion: Likelihood Ratio Test\t\t\t\t  ||"<<endl;    
    else if (mode=="C" && TMODE=="L" && PW_DATA) 
      outfile<<"||      Spliting criterion: Pairwise Likelihood Ratio Test\t\t\t\t  ||"<<endl;  
    outfile<<"||      Training dataset size = "<<train_size<<" "<<"(seed = "<<SEED; 
    if(CV_TEST)
      outfile<<", test stage = "<<TEST_STAGE;
    if(!CV_TEST)
      outfile<<", "<<TRAIN_PROP;
    outfile<<")"<<"\t\t\t\t\t  ||"<<endl;
    outfile<<"||      Validation dataset size = "<<val_size;
    if(!CV_TEST) 
      outfile<<" ("<<1.0-TRAIN_PROP<<")";
    outfile<<"\t\t\t\t\t  ||"<<endl;
    outfile<<"||      Maxmimum node size = "<<MIN_NODE_SIZE<<"         \t\t\t\t  ||"<<endl;
    outfile<<"||      Rank use to split = 1-"<<RANK_ITEM<<"         \t\t\t\t  ||"<<endl;
    outfile<<"||      Predicted Ranks in FOOT RULE = 1-"<<RANK_SIZE   <<"  \t\t\t\t  ||"<<endl;
    if (prediction == "F") 
      outfile<<"||      Prediction Mode = Most frequent ranking   \t\t\t\t  ||"<<endl;
    else if (prediction == "M") 
      outfile<<"||      Prediction Mode = Mean ranks   \t\t\t\t  ||"<<endl;
    else if (prediction == "C") 
      outfile<<"||      Prediction Mode = Modal ranking   \t\t\t\t  ||"<<endl;
    if (mode=="S") 
      outfile<<"||      Chisq value (Statistical test) = "<<st_val<< " (" << ST_ALP << ")" << "  \t\t\t\t\t  ||"<<endl;
    outfile<<"----------------------------------------------------------------------------"<<endl<<endl;
  }


  // Post: if x=0, print training data; if x=1, print testing data; if x=2, print training and testing data
  //       access the node, then print the result
  void printFinal(const vector <node_info>& nodeInfo, vector <pair_comp>& pairs,
    double dist_matrix[MAX_RANK_SIZE][MAX_RANK_SIZE], int x){  
    outfile<<endl;
    outfile<<"-----------------------------------------------------------------------------"<<endl;
    outfile<<"||                               LEAF  NODES                               ||"<<endl;
    outfile<<"-----------------------------------------------------------------------------"<<endl;

    for (int i=0 ; i<ITEM_SIZE ; i++)
      for (int j=i+1 ; j<ITEM_SIZE ; j++)
        pairs.push_back(pair_comp(i,j,0.0,0.0));
    for (int i=0 ; i<RANK_SIZE ; i++)
      for (int j=0 ; j<RANK_SIZE ; j++)
        dist_matrix[i][j] = 0.0;  

    for (size_t k=0,n=0 ; k<nodeInfo.size() ; k++)
      if (nodeInfo[k].fn == 1){                        
        double first_rank[MAX_ITEM_SIZE];
        double mean_rank[MAX_ITEM_SIZE];
        int size;      
        double count[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE];
        double val_count[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE];
        outfile<<"("<<++n<<") ";
        
        if (x==0){
          countData(count, nodeInfo[k].mbr);
          size = nodeInfo[k].mbr.size();
          accessNode1(nodeInfo[k].mbr, first_rank, mean_rank);
          vector<int> p_rank = accessNode2(nodeInfo[k].mbr, first_rank, mean_rank);
          accessNode3(count, dist_matrix, pairs, p_rank, first_rank, mean_rank, size);
          printNode1(first_rank, mean_rank, size, nodeInfo[k].id, p_rank);
          printNode2(count, size);
        }
        else if (x==1) {
          countData(count, nodeInfo[k].smbr);
          size = nodeInfo[k].smbr.size();      
          accessNode1(nodeInfo[k].smbr, first_rank, mean_rank);
          vector<int> p_rank = accessNode2(nodeInfo[k].smbr, first_rank, mean_rank);
          accessNode3(count, dist_matrix, pairs, p_rank, first_rank, mean_rank, size);
          printNode1(first_rank, mean_rank, size, nodeInfo[k].id, p_rank);
          printNode2(count, size);
        } 
        else if (x==2) { // assess performance based on prediction of best pruned tree using validation data
          countData(count, nodeInfo[k].mbr);
          countData(val_count, nodeInfo[k].smbr);
          size = nodeInfo[k].mbr.size();
          int val_size = nodeInfo[k].smbr.size();
          accessNode1(nodeInfo[k].mbr, first_rank, mean_rank);
          vector<int> p_rank = accessNode2(nodeInfo[k].mbr, first_rank, mean_rank);
          outfile << "Training ";
          printNode1(first_rank, mean_rank, size, nodeInfo[k].id, p_rank);
          if (val_size > 0){
            accessNode1(nodeInfo[k].smbr, first_rank, mean_rank);
            vector<int> p_rank_val = accessNode2(nodeInfo[k].smbr, first_rank, mean_rank);
            accessNode3(val_count, dist_matrix, pairs, p_rank, first_rank, mean_rank, val_size); 
            outfile << "  Validation "; 
            printNode1(first_rank, mean_rank, val_size, nodeInfo[k].id, p_rank_val);
            printNode2(val_count, val_size);
          }
          else if (val_size == 0){
            outfile<<"  Validation "<<nodeInfo[k].id<<"\t Size= "<<val_size<<"\t"<<endl;
            outfile<<" - -"<<endl;
          }
        } 
      }    
  }   


  // Post: print the result of performance assessment by "Discrepancy measure based on pair comparisons"  
  void printPref1(const vector <pair_comp>& pairs, int size){   
    outfile<<setprecision(5)<<endl;
    outfile<<"----------------------------------------------------------------------------"<<endl;
    outfile<<"||      Access method: Discrepancy measure based on pair comparisons      ||"<<endl;
    outfile<<"----------------------------------------------------------------------------"<<endl;
    outfile<<"                        |  Expect      Observe  |"<<endl;
    outfile<<"------------------------------------------------|"<<endl;
    for (size_t i=0 ; i<pairs.size() ; i++)
      outfile<<"| "<<qf_input.findTarget(pairs[i].item1)<<" is preferred to "<<qf_input.findTarget(pairs[i].item2)
        <<"\t|   "<<pairs[i].exp_p<<"     \t"<<pairs[i].obs_p<<"\t|"<<endl;
    double discrepancy1 = calculateDiscrepancy1(pairs, size);
    outfile<<"----------------------------------------------------------------------------"<<endl;
    outfile<<"  Discrepancy           = "<<discrepancy1<<endl;
    outfile<<"----------------------------------------------------------------------------"<<endl;
  }


  // Post: print the result of performance accessment by "Foot Rule"   
  void printPref2(const vector <double>& alpha, double dist_matrix[MAX_RANK_SIZE][MAX_RANK_SIZE], int size){ 
    vector<double> v0;
    for (int i=0 ; i<RANK_SIZE ; i++)
      v0.push_back(i+1);
    if (ITEM_SIZE != RANK_SIZE)   
      v0.push_back((ITEM_SIZE+RANK_SIZE+1)*1.0/2);
    double max_discrepancy = ut.maxDiscrepancy(alpha, v0);
                 
    outfile<<setprecision(5)<<endl;
    outfile<<"----------------------------------------------------------------------------"<<endl;
    outfile<<"||      Access method: Foot Rule                                          ||"<<endl;
    outfile<<"----------------------------------------------------------------------------"<<endl;
    outfile<<"\t\t\t    |     -- Actual Rank --"<<endl;
    outfile<<"\t\t\t    |\t";
    for (int i=0 ; i<RANK_SIZE ; i++)
      outfile<<" "<<(i+1)<<"\t";
    outfile<<"other"<<endl;
    outfile<<"------------------------------------------------------------------|"<<endl;

    double dist_matrix2[MAX_RANK_SIZE+1][MAX_RANK_SIZE+1]; 
    for (int i=0 ; i<RANK_SIZE ; i++)
      for (int j=0 ; j<RANK_SIZE ; j++) 
        dist_matrix2[i][j] = dist_matrix[i][j];
              
    double cnt2 = 0.0;
    for (int i=0 ; i<RANK_SIZE ; i++){
      outfile<<"| Predicted Rank - "<<i+1<<"\t    |";
      double cnt1 = 0.0;
      for (int j=0 ; j<RANK_SIZE ; j++){    
        outfile<<"\t"<<dist_matrix[i][j];  // fill in the row of predicted "rank i" with different actual "rank j"
        cnt1 += dist_matrix[i][j];
      }
      cnt2 += (size*1.0 - cnt1);  // column sum of the actual "rank other"
      outfile<<"\t"<<(size*1.0 - cnt1)<<endl;  // fill in the cell of predicted "rank i" and actual "rank other"
      dist_matrix2[i][RANK_SIZE] = (size*1.0 - cnt1);  // assign the RHS column of dist_matrix2
    }
    outfile<<"| Predicted Rank - Other    |";
    for (int i=0 ; i<RANK_SIZE ; i++){ 
      double cnt3 = 0.0; 
      for (int j=0 ; j<RANK_SIZE ; j++)  
        cnt3 += dist_matrix[j][i];  // column sum of the actual "rank i"
      outfile<<"\t"<<(size*1.0 - cnt3);  // fill in the cell of predicted "rank other" and actual "rank i"
      dist_matrix2[RANK_SIZE][i] = (size*1.0 - cnt3);  //assign the bottom row of dist_matrix2
    }    
    if (RANK_SIZE != ITEM_SIZE){
      outfile<<"\t"<<((ITEM_SIZE-RANK_SIZE)*size*1.0 - cnt2)<<endl;  
      dist_matrix2[RANK_SIZE][RANK_SIZE] = ((ITEM_SIZE-RANK_SIZE)*size*1.0 - cnt2); 
    }
    else{ 
      outfile<<"\t0"<<endl;  
      dist_matrix2[RANK_SIZE][RANK_SIZE] = 0; 
    }
          
    double discrepancy2 = calculateDiscrepancy2(dist_matrix2, alpha, size);
    outfile<<"----------------------------------------------------------------------------"<<endl;
    outfile<<"  Alpha                 = {";
    for (size_t i=0 ; i<alpha.size()-1 ; i++)
      outfile<<alpha[i]<<",";
    outfile<<alpha[alpha.size()-1]<<"}"<<endl;
    outfile<<"  Maximunm Discrepancy  = "<<max_discrepancy<<endl;
    outfile<<"  Discrepancy           = "<<discrepancy2<<endl;
    outfile<<"  Normalize Discrepancy = "<<1 - discrepancy2/max_discrepancy<<endl;
    outfile<<"----------------------------------------------------------------------------"<<endl<<endl;
  }


  // Pre: after CART cross-validation, input all the nodes of the final tree
  // Post: print the final tree, return the miss. rate
  double printFinalTree(vector <node_info> nodeInfo){       
    double corrp = 0.0;
    double total = 0.0;     
           
    for (size_t n=0 ; n<nodeInfo.size() ; n++){
      double first_rank[MAX_ITEM_SIZE];
      double mean_rank[MAX_ITEM_SIZE];          
      double count[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE];

      countData(count, nodeInfo[n].mbr);
      int size = nodeInfo[n].mbr.size();
      accessNode1(nodeInfo[n].mbr, first_rank, mean_rank);
      vector<int> p_rank = accessNode2(nodeInfo[n].mbr, first_rank, mean_rank);    
      printNode1(first_rank, mean_rank, size, nodeInfo[n].id, p_rank);
      printNode2(count, size);
       
      if (nodeInfo[n].fn == 0){ 
        if (nodeInfo[n].st == 0)
          outfile<<"Split var: "<<nodeInfo[n].sv0<<", Split val: "<<qf_input.getDescription(nodeInfo[n].sv0)
              <<" < "<<nodeInfo[n].sv1<<" ==> Node "<<nodeInfo[n].id<<"1; otherwise, ==> Node "
              <<nodeInfo[n].id<<"2";
        else{
          outfile<<"Split var: "<<nodeInfo[n].sv0<<", Split val: "<<qf_input.getDescription(nodeInfo[n].sv0)<<" = ";
          for (size_t i=0 ; i<nodeInfo[n].sv2.size() ; i++)
            outfile<<qf_input.findInput(nodeInfo[n].sv0, (nodeInfo[n].sv2)[i])<<", ";   
          outfile<<" ==> Node "<<nodeInfo[n].id<<"1; otherwise, ==> Node "<<nodeInfo[n].id<<"2";  
        }
      }
      else{
        switch (RANK_SIZE){  //calculate top-k misclassification rate for the final pruned tree 
          case 1:   
            for (int i=0 ; i<ITEM_SIZE ; i++)
              for (int j=0 ; j<ITEM_SIZE ; j++)
                corrp += count[qf_input.find(p_rank[0])][i][j];
            break;
          case 2:  
            for (int i=0 ; i<ITEM_SIZE ; i++)
              corrp += count[qf_input.find(p_rank[0])][qf_input.find(p_rank[1])][i];
            break;  
          case 3:   
            corrp += count[qf_input.find(p_rank[0])][qf_input.find(p_rank[1])][qf_input.find(p_rank[2])];
            break;
        }     
        total += size;
        nodeInfo[n].fn = 1;      
        outfile<<" ******************* ";
      }
      outfile<<endl<<endl;
    }    
    return (1.0-1.0*corrp/total);
  }


  // Pre: finished all stage of cross-validation
  // Post: print the performance of all trees, choose the best tree
  vector <node_info> printCartFinal(const vector <vector <node_info> >& cartNodeInfo, const vector <tree_info>& cartTreeInfo,
    int train_size, vector <Data> train, int val_size, vector <Data> valid){
    outfile<<"----------------------------------------------------------------------------"<<endl;
    outfile<<"||      Tree Result (Final)                                               ||"<<endl;
    outfile<<"----------------------------------------------------------------------------"<<endl;
    outfile<<"| k\t|T|\tR(T)\tR^(T)\tS.E.\talpha\t|"<<endl;
    outfile<<"|------------------------------------------------"<<setprecision(4)<<endl;
    
    int bestTree = -1;
    int leaveNum = 0;
    double missClass = MAX_VAL;
    double s0 = 0.0;
    
    for (size_t n=0 ; n<cartTreeInfo.size() ; n++){  // calculate R^(CV) and its SE for each pruned subtree
      double m = ut.mean(cartTreeInfo[n].er);        // T^(k) in training stage (-1)
      double s = sqrt(ut.standardError(cartTreeInfo[n].er,m)/CART_SIZE);
      if (n == 0)
        s0 = s;
      outfile<<"| "<<cartTreeInfo[n].id<<"\t"<<cartTreeInfo[n].size<<"\t"
          <<cartTreeInfo[n].mi<<"\t"<<m<<"\    t"<<s<<"\t"<<cartTreeInfo[n].ap<<endl;
      if (m <= missClass){
        bestTree = n;
        leaveNum = cartTreeInfo[n].size;
        missClass = m;
        s0 = s;
      } 
      if (m <= missClass + s0){  // 1 SE Rule
        bestTree = n;
        leaveNum = cartTreeInfo[n].size;
      } 
    }

    vector <node_info> bestNodeInfo;
    bestNodeInfo = cartNodeInfo[bestTree];
    // put validation data in the best pruned subtree
    for (size_t n=0 ; n<bestNodeInfo.size() ; n++)
      bestNodeInfo[n].smbr.clear();
    putTestData(bestNodeInfo, valid);
      
    outfile<<"----------------------------------------------------------------------------"<<endl;
    outfile<<"||      Best Tree = Tree #"<<(bestTree+1)<<"\t\t\t\t\t\t  ||"<<endl;
    outfile<<"||      Number of Leave = "<<leaveNum<<"\t\t\t\t\t\t  ||"<<endl;
    outfile<<"----------------------------------------------------------------------------"<<endl;
    outfile<<"||      Best pruned subtree based on training data                        ||" << endl;
    outfile<<"----------------------------------------------------------------------------"<<endl;
    double missRate = printFinalTree(cartNodeInfo[bestTree]);  
    outfile<<"----------------------------------------------------------------------------"<<setprecision(4)<<endl;
    outfile<<"||      Misclassification Cost = "<<missClass<<"\t\t\t\t\t  ||"<<endl;
    outfile<<"||      Misclassification Rate = "<<missRate<<"\t\t\t\t\t  ||"<<endl;  
    outfile<<"----------------------------------------------------------------------------"<<endl;

    
    double dist_matrix[MAX_RANK_SIZE][MAX_RANK_SIZE];
    vector <pair_comp> pairs; 
    outfile<<"----------------------------------------------------------------------------"<<endl;
    outfile<<"||      Best pruned subtree based on training data                        ||";
    printFinal(bestNodeInfo,pairs,dist_matrix,0);  /** print leaves of best pruned subtree based on training data **/
    printPref1(pairs,train_size);                  /** access performance using paired comparison **/
    printPref2(alpha,dist_matrix,train_size);      /** access performance using footrule discrepancy **/    

    if (val_size>0){
      double dist_matrix_val[MAX_RANK_SIZE][MAX_RANK_SIZE];
      vector <pair_comp> pairs_val; 
      outfile<<"----------------------------------------------------------------------------"<<endl;
      outfile<<"||      Best pruned subtree based on validation data                      ||";
      printFinal(bestNodeInfo,pairs_val,dist_matrix_val,2);   /** print leaves of best pruned subtree based on validation data **/
      printPref1(pairs_val,val_size);                    /** access performance using paired comparison **/
      printPref2(alpha,dist_matrix_val,val_size);        /** access performance using footrule discrepancy **/
    }
    return bestNodeInfo;
  }


  // Print the node info during pruning
  void printPruning(const vector <node_info>& nodeInfo, int tree_n, int node_size, int final_node_size){
    outfile<<"Tree #00"<<tree_n<<", Num of Nodes = "<<node_size<<", Num of Final Nodes = "<<final_node_size<<endl;    
    outfile<<"----------------------------------------------------------------------------"<<endl;     
    for (int n=nodeInfo.size()-1 ; n>-1 ; n--){
      int cnt = 0;
      for (size_t m=0 ; m<nodeInfo.size() ; m++)
        if (nodeInfo[m].fn==1 && nodeInfo[n].id.length()<nodeInfo[m].id.length() 
            && nodeInfo[n].id==nodeInfo[m].id.substr(0,nodeInfo[n].id.length())) 
          cnt++;
      if (nodeInfo[n].fn == -1)      /* pruned node */
        outfile<<" -- "<<nodeInfo[n].id<<", rv:"<<nodeInfo[n].rv<<", gv:"<<nodeInfo[n].gv<<", size:"<<(nodeInfo[n].mbr).size()<<":"<<(nodeInfo[n].smbr).size()<<endl;
      else if (nodeInfo[n].fn == 0)  /* internal node */
        outfile<<" @"<<cnt<<" "<<nodeInfo[n].id<<", rv:"<<nodeInfo[n].rv<<", gv:"<<nodeInfo[n].gv<<", size:"<<(nodeInfo[n].mbr).size()<<":"<<(nodeInfo[n].smbr).size()<<endl; 
      else if (nodeInfo[n].fn == 1)  /* terminal node */
        outfile<<" ## "<<nodeInfo[n].id<<", rv:"<<nodeInfo[n].rv<<", gv:"<<nodeInfo[n].gv<<", size:"<<(nodeInfo[n].mbr).size()<<":"<<(nodeInfo[n].smbr).size()<<endl;  
    }
    outfile<<"----------------------------------------------------------------------------"<<endl<<endl;
  }


  // Pre: Error in parameter
  // Post: Display the usage
  void errorHandling(){
    outfile<<"Usage:"<<endl;
    outfile<<"<default> [Use Gini] [Max Node Size = 300] [Rank used = 1-3]"<<endl;
    outfile<<"          [Chi-square value = 0.01]"<<endl;
    outfile<<"dtree -T [E:Entropy; G:Gini; C:Chi-square Test; S:Statistical Test] -S [Max Node Size]"<<endl;
    outfile<<"      -I [Rank used] -C [alpha value] > [Output File Name]"<<endl;
    // exit(0);
  }

//*****************************************************************************************************************

void mytest()
{
  outfile << ITEM_SIZE << " " << RANK_SIZE << " " << CART_SIZE << endl;
  outfile << DATA_FILE << endl;
  outfile << INFO_FILE << endl;
  outfile << ST_ALP << " " << CHI_ALP << endl;
  outfile << TRAIN_PROP << endl;
  outfile << mode << " " << TMODE << " " <<
    pairwise << " " << PW_DATA << " " << prediction << 
    " " << NODE_SIZE << " " << MIN_NODE_SIZE << " " <<
    RANK_ITEM << " " << CV_TEST << endl;
}


List getTreeList(vector <node_info> nodeInfo)
{
  // Created by Li Qinglong
  // Aim to output the nodeInfo of the tree to R
  // similar to printFinalTree function
  List nodeInfoList(nodeInfo.size());
  double corrp = 0.0;
  double total = 0.0;     
         
  for (size_t n=0 ; n<nodeInfo.size() ; n++)
  {
    double first_rank[MAX_ITEM_SIZE];
    double mean_rank[MAX_ITEM_SIZE];          
    double count[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE];

    countData(count, nodeInfo[n].mbr);
    int size = nodeInfo[n].mbr.size();
    accessNode1(nodeInfo[n].mbr, first_rank, mean_rank);
    vector<int> p_rank = accessNode2(nodeInfo[n].mbr, first_rank, mean_rank);    

    // convert p_rank into string
    stringstream ss;
    ss << "[" << p_rank[0];
    for (size_t i = 1; i < p_rank.size(); ++i)
    ss << "," << p_rank[i];
    ss << "]";
    string str_p_rank;
    ss >> str_p_rank;
    // get split value string, "" if leaf node
    vector<double> vec_split_value;
    if (nodeInfo[n].fn == 0)
    {
      if (nodeInfo[n].st == 0)
        vec_split_value.push_back(nodeInfo[n].sv1);
      else
        for (size_t i=0; i<nodeInfo[n].sv2.size(); i++)
          vec_split_value.push_back(qf_input.findInput(nodeInfo[n].sv0, (nodeInfo[n].sv2)[i]));
    }
    else
    {
      vec_split_value.push_back(NA_REAL);
    }
    NumericVector split_value(vec_split_value.begin(), vec_split_value.end());
    nodeInfoList[n] = List::create(
              Named("Node_id") = nodeInfo[n].id,
              Named("Size") = size,
              Named("predicted_rank") = str_p_rank,
              Named("Split_var_id") = nodeInfo[n].sv0,
              Named("Split_var_name") = qf_input.getDescription(nodeInfo[n].sv0),
              Named("Split_value") = split_value,
              Named("isLeafNode") = nodeInfo[n].fn,
              Named("isDiscrete") = nodeInfo[n].st                   
              );

  } // end of for
  return nodeInfoList;
}



//' @export

// [[Rcpp::export]]
List decTreeImpFuncCpp(
  const int myITEM_SIZE,
  const int myRANK_SIZE,
  const int myCART_SIZE,
  const std::string myDATA_FILE,
  const std::string myINFO_FILE,
  const std::string myOUTPUT_FILE,
  const std::string mymode,
  const std::string myTMODE,
  const bool mypairwise,      
  const bool myPW_DATA,
  const std::string myprediction,  
  const int myNODE_SIZE,      
  const int myMIN_NODE_SIZE,  
  const int myRANK_ITEM,
  const double myST_ALP,      
  const double myCHI_ALP,     
  const double myTRAIN_PROP,  
  const bool myCV_TEST,       
  const int myTEST_STAGE  
  ){
  // get the inputs
  ITEM_SIZE = myITEM_SIZE;
  RANK_SIZE = myRANK_SIZE;
  CART_SIZE = myCART_SIZE;
  DATA_FILE = myDATA_FILE;
  INFO_FILE = myINFO_FILE; 
  outfile.open(myOUTPUT_FILE.c_str());
  mode = mymode;
  TMODE = myTMODE;
  pairwise = mypairwise;
  PW_DATA = myPW_DATA;
  prediction = myprediction;
  NODE_SIZE = myNODE_SIZE;
  MIN_NODE_SIZE = myMIN_NODE_SIZE;
  RANK_ITEM = myRANK_ITEM;
  ST_ALP = myST_ALP;
  CHI_ALP = myCHI_ALP;
  TRAIN_PROP = myTRAIN_PROP;
  CV_TEST = myCV_TEST;
  TEST_STAGE = myTEST_STAGE;

  // end of getting of the inputs
  vector <Data> allData;                     /* whole set of data */
  vector <Data> allData_cp;                  /* copy of whole dataset */
  vector <Data> trainData;                   /* training set of data */
  vector <Data> valData;                     /* validation set of data */
  vector <Data> cartData [10];               /* each contain ~1/10 of trainData */
  vector <Question> que;                     /* all the variables */
  vector <vector <node_info> > cartNodeInfo; /* nodes of nested seq. of T0, t1,...Tk */
  vector <tree_info> cartTreeInfo;           /* info nested seq. of T0, T1,...Tk */
  vector <double> cartAlpha;                 /* values sqrt(alpha[k][k+1]) of T0, T1,...Tk */
  vector <Data> dummyDataSet;
  vector <int> dummy;
  vector <double> dummy2;
  double CHI_VAL; // Chi-square test critical value
  double ST_VAL; // Statistical test critical value
  int stage = -1;
   
  
  qf_input.init(allData, que, alpha, INFO_FILE, DATA_FILE, RANK_SIZE, ITEM_SIZE);

  /* data partition of whole data set */
  if (CV_TEST){
    // mysrand() and myrand() are same as srand() and rand() in cstdlib
    mysrand(SEED);
    int rand_int;
    int datasize = allData.size();
    for(int n=0; n<datasize; n++){ 
      rand_int = myrand()%(datasize-n); 
      allData_cp.push_back(allData[rand_int]); // construct training data by random sampling
      allData.erase(allData.begin()+rand_int);
    }
    for (unsigned i=0 ; i<allData_cp.size() ; i++){ // for 10-fold cross validation testing
      if (i%10 == TEST_STAGE)
        valData.push_back(allData_cp[i]);
      else
        trainData.push_back(allData_cp[i]);
    }
   // for (int i=0 ; i<valData.size() ; i++)
   //   outfile<<valData[i].getID()<<endl;
  }
  else if (!CV_TEST){
    for (unsigned i=0 ; i<allData.size() ; i++)
      allData_cp.push_back(allData[i]);
    mysrand(SEED);
    int rand_int; 
    for(unsigned n=0; n<floor(TRAIN_PROP*allData.size()+0.5); n++){ 
      rand_int = myrand()%(allData.size()-n); 
      trainData.push_back(allData_cp[rand_int]); // construct training data by random sampling
      allData_cp.erase(allData_cp.begin()+rand_int);
    }
    for (unsigned i=0 ; i<allData_cp.size() ; i++)
      valData.push_back(allData_cp[i]);  
  }    

  for (unsigned i=0 ; i<trainData.size() ; i++) //partition data for cross-validation by systematic sampling
    cartData[i%CART_SIZE].push_back(trainData[i]);

  /* extract parameters */ 
  /*
  if (argc%2 != 1)
    errorHandling();        
  for (int i=1 ; i<argc ; i+=2){
    string opt = argv[i];  
    if (opt == "-T"){
      if (string(argv[i+1])== "E" || string(argv[i+1])== "G" || string(argv[i+1])== "C" || string(argv[i+1])== "S")
        mode = string(argv[i+1]);
      else errorHandling();  
    }
    else if (opt == "-S"){   
      MIN_NODE_SIZE = atoi((string(argv[i+1])).c_str());
      NODE_SIZE = MIN_NODE_SIZE;
    }  
    else if (opt == "-I")  
      RANK_ITEM = atoi((string(argv[i+1])).c_str());
    else if (opt == "-C")
      ST_ALP = atof((string(argv[i+1])).c_str());  
    else
      errorHandling();
  }
  */
  /* print herder of report */
  ST_VAL = ut.chiSqrValue(ST_ALP, ITEM_SIZE-1);
  CHI_VAL = ut.chiSqrValue(CHI_ALP, 1);
  printHeader(trainData.size(), valData.size(), ST_VAL);  

  /******** do while all subset have finished processing ********/
  while (stage < 10){
    
    vector <Data> dataSet0;
    vector <Data> testData;
    queue <vector <Data> > currNodes;      // store the nodes that haven't check
    queue <string> currNodesID;            // store the id of nodes that haven't check
    vector <node_info> nodeInfo;
    vector <tree_info> treeInfo;  
     
    if (stage == -1){
      dataSet0 = trainData; 
      testData = trainData;
    }   
    else{
      for (int k=0 ; k<CART_SIZE ; k++)
        if (k != stage)
          for (unsigned i=0 ; i<cartData[k].size() ; i++)
            dataSet0.push_back(cartData[k][i]);
      testData = cartData[stage]; 
    } 
    
    /* // testing
    double testing1[ITEM_SIZE][ITEM_SIZE][ITEM_SIZE], testing2;
    countData(testing1, trainData);
    testing2=getValue(trainData.size(), 3, testing1);
    outfile<<"testing2:";
    outfile<<testing2<<endl;
    // testing end */
    
    MIN_NODE_SIZE = NODE_SIZE * dataSet0.size() / trainData.size();   
                  
    currNodes.push(dataSet0);
    currNodesID.push("0."); 
    nodeInfo.push_back(node_info(dataSet0,dummyDataSet,"0.",0.0,0.0,0,0,0,0.0,dummy));  
                   
    /******** do while all nodes have finished splitting ********/
    while (currNodes.size() != 0){
      // get a node, count data, and calculate initial entropy
      vector <Data> node = currNodes.front();
      currNodes.pop(); //remove the top element of currNodes
      string id = currNodesID.front();
      currNodesID.pop();
        
      if (node.size() == 0){
        if (stage == -1)      
          outfile<<"** ** **"<<endl<<" - - - - - - - - "<<endl;
        continue;
      }
    
      bool enhance = false;
      int split_var = -1;
      vector <int> split_val;
      double split_val2 = 0.0;
      double count[MAX_ITEM_SIZE][MAX_ITEM_SIZE][MAX_ITEM_SIZE]; 
      countData(count, node);
      double curr_value;
      double new_value;
      vector <vector <double> > invC; 
      if (mode == "E" || mode == "G"){
        curr_value = getValue(node.size(), RANK_ITEM, count); // calculate impurity of parent node
        if (stage == -1)
          outfile<<"Initial entropy/gini of node '"<<id<<"' = "<<setprecision(6)<<curr_value<<endl;
      }
      else if (mode == "S" || mode == "C"){
        if (stage == -1)
          outfile<<"Node '"<<id<<"'"<<endl; 
        if (mode == "S"){
          invC = getMatInv(node);
          curr_value = -ST_VAL; // set threshold value for significant splits for statistical test
        }
        if (mode == "C")
          curr_value = -CHI_VAL; // set threshold value for significant splits for chi-square test
        }
      // check all possible variable
      for (size_t sv=0 ; sv<que.size() ; sv++){
        // discrete variable
        if (que[sv].getType() == "d"){
          vector <vector <int> > possible_split = ut.gen_com(que[sv].getSize());

          // check all possible binary split
          for (size_t ps=0 ; ps<possible_split.size() ; ps++){
            // assign data to nodes 
            vector <Data> dataSet1 , dataSet2;        
            for (size_t k=0 ; k<node.size() ; k++){
              double x = qf_input.find(sv, node[k].getInput(sv));
              bool match = false;
              for (size_t i=0 ; !match&&i<possible_split[ps].size() ; i++)
                if (possible_split[ps][i] == x)
                  match = true;
              if (match)
                dataSet1.push_back(node[k]);
              else dataSet2.push_back(node[k]); 
            } 
            // calculate entropy

            new_value = newValue(dataSet1, dataSet2, invC);                 
            if (new_value <= curr_value-0.0000001){
              curr_value = new_value;
              split_var = sv;
              split_val = possible_split[ps];
              enhance = true;
            }
          } // END for all possible split   
        } // END if discrete variable  
            
        // continuous variable  
        else{ 
          double x1 = qf_input.findInput(sv, 0);
          double x2 = qf_input.findInput(sv, 1);
          int x3 = que[sv].getSize();        // num of split value 
          for (int i=1 ; i<=x3 ; i++){
            double y = x1 + (x2-x1)*1.0*i/x3;
            vector <Data> dataSet1, dataSet2;        
            for (size_t k=0 ; k<node.size() ; k++){
              double x = node[k].getInput(sv);
              if (x < y)
                dataSet1.push_back(node[k]);
              else dataSet2.push_back(node[k]); 
            }         
            // calculate entropy
            
            new_value = newValue(dataSet1, dataSet2, invC);                         
            if (new_value < curr_value){
              curr_value = new_value;
              split_var = sv;
              split_val2 = y;
              enhance = true;
            }     
          } // END for possible split  

        } // END continuous variable             
      } // END for all split variable 
       
      if (enhance){ //case if there is enhancement after node split
        // the best split, assign data to nodes 
        vector <Data> dataSet1 , dataSet2; 
        int split_type;
        double split_value1 = 0.0;
        vector <int> split_value2;    
        int split_vari = split_var;  
        // discrete variable
        if (que[split_var].getType() == "d"){
          split_type = 1;                           
          for (size_t k=0 ; k<node.size() ; k++){
            double x = qf_input.find(split_var, node[k].getInput(split_var));
            bool match = false;
            for (size_t i=0 ; !match&&i<split_val.size() ; i++)
              if (split_val[i] == x)
                match = true; 
            if (match)
              dataSet1.push_back(node[k]);
            else dataSet2.push_back(node[k]); 
          }
          if (stage == -1)
            outfile<<"Split var: "<<split_var<<", Split val: "<<qf_input.getDescription(split_var)<<" = ";
          for (size_t i=0 ; i<split_val.size() ; i++){
            if (stage == -1)
              outfile<<qf_input.findInput(split_var, split_val[i])<<", ";
            split_value2.push_back(split_val[i]);
          }
        }
        // continuous variable
        else{
          split_type = 0;
          for (size_t k=0 ; k<node.size() ; k++){
            double x = node[k].getInput(split_var);
            if (x < split_val2)
              dataSet1.push_back(node[k]);
            else dataSet2.push_back(node[k]); 
          }
          split_value1 = split_val2;
          if (stage == -1)
            outfile<<"Split var: "<<split_var<<", Split val: "<<qf_input.getDescription(split_var)<<" < "<<split_val2<<", ";
        }   
        if (stage == -1){
          if (mode == "S" || mode == "C"){
            outfile<<"New Value = "<<setprecision(6)<<curr_value*(-1.0);
            if (mode == "C"){
              int deg;
              deg = df(dataSet1, dataSet2);
              outfile << "(df = " << deg <<", ";
              outfile << "chi-square CV = " << ut.chiSqrValue(CHI_ALP,deg)<<")";
            }
            outfile<<endl;
          }
          else if (mode == "E" || mode == "G")
            outfile<<"New Value = "<<curr_value<<endl;
        }
        string tmp1 = id + "1";
        string tmp2 = id + "2";
        double first_rank[MAX_ITEM_SIZE];
        double mean_rank[MAX_ITEM_SIZE];      
    
        accessNode1(dataSet1, first_rank, mean_rank);
        if (stage == -1)
          printNode1(first_rank, mean_rank, dataSet1.size(), tmp1, dummy);
        accessNode1(dataSet2, first_rank, mean_rank);
        if (stage == -1)
          printNode1(first_rank, mean_rank, dataSet2.size(), tmp2, dummy);      
        if (stage == -1)  
          outfile<<" - - - - - - - - "<<endl;
       
        if (dataSet1.size()>0 && dataSet1.size() <= (size_t)MIN_NODE_SIZE){
          nodeInfo.push_back(node_info(dataSet1,dummyDataSet,tmp1,0.0,0.0,1,0,0,0.0,dummy));           
          currNodes.push(dummyDataSet);
          currNodesID.push("");
        }
        else if (dataSet1.size()>0){
          nodeInfo.push_back(node_info(dataSet1,dummyDataSet,tmp1,0.0,0.0,0,0,0,0.0,dummy));  
          currNodes.push(dataSet1);
          currNodesID.push(tmp1);
        }      
        if (dataSet2.size()>0 && dataSet2.size() <= (size_t)MIN_NODE_SIZE){
          nodeInfo.push_back(node_info(dataSet2,dummyDataSet,tmp2,0.0,0.0,1,0,0,0.0,dummy));  
          currNodes.push(dummyDataSet);
          currNodesID.push("");  
        }
        else if (dataSet2.size()>0){
          nodeInfo.push_back(node_info(dataSet2,dummyDataSet,tmp2,0.0,0.0,0,0,0,0.0,dummy));         
          currNodes.push(dataSet2);
          currNodesID.push(tmp2);
        }  
        // update split variable, value
        for (size_t n=0 ; n<nodeInfo.size() ; n++)
          if (nodeInfo[n].id == id){
            nodeInfo[n].st = split_type;
            nodeInfo[n].sv0 = split_vari;
            nodeInfo[n].sv1 = split_value1;
            nodeInfo[n].sv2 = split_value2;
            break;  
          }        
             
      } // END enhance
      else{
        if (stage == -1)
        {
          if (mode != "S" && mode != "C")
            outfile<<"stop split (no enhancement)  New Value = "<<new_value<<"= "<<"previous impurity"<<curr_value
                <<endl<<" - - - - - - - - "<<endl;  
          else if (mode == "S" || mode == "C")
            outfile<<"stop split (not significant) New Value = "<<new_value<<" < critical value"<<curr_value
            <<endl<<" - - - - - - - - "<<endl;      
        }
        for (size_t n=0 ; n<nodeInfo.size() ; n++)
          if (nodeInfo[n].id == id) 
            nodeInfo[n].fn = 1;      
      }      
    } // END while /** finished all splitting **/
    
    /* print final node of unpruned tree, access performance */
    double dist_matrix[MAX_RANK_SIZE][MAX_RANK_SIZE];
    vector <pair_comp> pairs; 
    if (stage == -1){
      printFinal(nodeInfo,pairs,dist_matrix,0);             /** print final result (unpruned tree) **/
      printPref1(pairs,dataSet0.size());                    /** access performance (unpruned tree) **/
      printPref2(alpha,dist_matrix,dataSet0.size());        /** access performance (unpruned tree) **/    
    }       

    /* put testing data */ 
    if (stage == -1){
      for (size_t n=0 ; n<nodeInfo.size() ; n++)
        nodeInfo[n].smbr = nodeInfo[n].mbr;
    }
    else {
      putTestData(nodeInfo, testData);  
    }
      
    //for (size_t n=0 ; n<nodeInfo.size() ; n++)
    //  outfile<<nodeInfo[n].id<<"\t: "<<nodeInfo[n].mbr.size()<<",\t"<<nodeInfo[n].smbr.size()<<endl;   
    //pa                           irs.clear();        
    //printFinal(nodeInfo,pairs,dist_matrix,1);            /** print final result (test data) **/
    //accessPerf(alpha,pairs,dist_matrix,testData.size()); /** access performance (test data) **/    
    
    /******** pruning of tree ********/  
    int final_node_size = 0;
    int node_size = nodeInfo.size();
    int tree_n = 0;
    double min_gv = 0; 
    for (int n=0 ; n<node_size ; n++)
      if (nodeInfo[n].fn == 1)
        final_node_size++;      
    
    /******** do while all pruning not ended ********/         
    while (true){       
      double all_rv = 0.0;
      tree_n ++;  // id of the pruned tree in each stage
  
      for (int n=nodeInfo.size()-1 ; n>-1 ; n--){  // check all nodes from bottom to top
        if (nodeInfo[n].fn == -1)  // skip to next node if nodeInfo[n] is a pruned node
           continue;            
        /* calculate r-value for each node*/             
        double discrepancy2 = 0.0; 
        if ((nodeInfo[n].smbr).size() != 0)
          discrepancy2 = accessNode6(nodeInfo[n]); 
        nodeInfo[n].rv = discrepancy2;
 
        /* calculate g-value for each internal node*/ 
        if (nodeInfo[n].fn == 0){
          int cnt = 0;
          double cnt2 = 0.0;
          for (size_t m=0 ; m<nodeInfo.size() ; m++)
            if (nodeInfo[m].fn==1 && nodeInfo[n].id.length()<nodeInfo[m].id.length() 
                && nodeInfo[n].id==nodeInfo[m].id.substr(0,nodeInfo[n].id.length())){  
              cnt++;  // count the no. of terminal nodes in the subtree nodeInfo[n]
              cnt2 += nodeInfo[m].rv;
            }  
          nodeInfo[n].gv = (nodeInfo[n].rv-cnt2)/(1.0*(cnt-1));
        } 
      }
    
      /* sum up R-value (misclassification-rate) */
      for (size_t n=0 ; n<nodeInfo.size() ; n++)
        if (nodeInfo[n].fn == 1)
          all_rv += nodeInfo[n].rv;  
      if (stage == -1)
        min_gv /= 10.0;                
      treeInfo.push_back(tree_info(tree_n,min_gv,final_node_size,all_rv,dummy2));
    
      if (stage == -1){  
        vector <node_info> tmpNodeInfo;
        for (size_t n=0 ; n<nodeInfo.size() ; n++)
          if (nodeInfo[n].fn != -1)   
            tmpNodeInfo.push_back(nodeInfo[n]);
        cartNodeInfo.push_back(tmpNodeInfo);
        cartTreeInfo.push_back(treeInfo.back());
      }
      // printPruning(nodeInfo, tree_n, node_size, final_node_size);
      /* stop pruning if reached the root node */
      if (final_node_size == 1)
        break;
      /* choose the node to be pruned */
      int prune_n = -1;
      min_gv = MAX_VAL;
      for (int n=nodeInfo.size()-1 ; n>-1 ; n--)
        if (nodeInfo[n].fn==0 && nodeInfo[n].gv<min_gv){
          prune_n = n;  // record the id of the node which has the smallest g-value
          min_gv = nodeInfo[n].gv; 
        }        
 
      /* update attribute of related nodes */
      /* 1. node(s) that pruned */
      for (int n=nodeInfo.size()-1 ; n>-1 ; n--)
        if (nodeInfo[prune_n].id.length()<nodeInfo[n].id.length() &&
            nodeInfo[prune_n].id==nodeInfo[n].id.substr(0,nodeInfo[prune_n].id.length())){     
          if (nodeInfo[n].fn == 1)
            final_node_size--;
          if (nodeInfo[n].fn != -1)  
            node_size --;
          nodeInfo[n].fn = -1;
          nodeInfo[n].rv = 0.0;
          nodeInfo[n].gv = 0.0;
        } 
  
      final_node_size++;
      /* 2. node that change from internal node to terminal node */
      nodeInfo[prune_n].fn = 1;
      nodeInfo[prune_n].gv = 0.0;      

    } // end while pruning not ended
    /**/

    outfile<<"----------------------------------------------------------------------------"<<endl;
    outfile<<"|| Stage = "<<stage<<"\t\t\t\t\t\t\t\t  ||"<<endl;
    outfile<<"|| Size of Training, Testing Data = ("<<dataSet0.size()<<","<<testData.size()<<")\t\t\t\t  ||"<<endl;
    outfile<<"----------------------------------------------------------------------------"<<endl;
    outfile<<"| k\t|T|\tR(T)\talpha\t\t|"<<endl;
    outfile<<"|----------------------------------------"<<setprecision(5)<<endl;

    for (size_t n=0 ; n<treeInfo.size() ; n++)    
      outfile<<"| "<<treeInfo[n].id<<"\t"<<treeInfo[n].size<<"\t"
          <<treeInfo[n].mi<<"\t"<<treeInfo[n].ap<<endl;
    outfile<<"|---------------------------------------|"<<endl;
    /**/   
    if (stage == -1){  // calculating alpha for tree using all training data
      cartAlpha.push_back(-(trainData.size()*1.0));         
      for (size_t k=1 ; k<cartTreeInfo.size()-1 ; k++){
        if (cartTreeInfo[k].ap<0 && cartTreeInfo[k+1].ap<0)
          cartAlpha.push_back(-sqrt(cartTreeInfo[k].ap*cartTreeInfo[k+1].ap)); 
        else if (cartTreeInfo[k].ap>=0 && cartTreeInfo[k+1].ap>=0)  
          cartAlpha.push_back(sqrt(cartTreeInfo[k].ap*cartTreeInfo[k+1].ap)); 
        else
          cartAlpha.push_back(-sqrt(-cartTreeInfo[k].ap*cartTreeInfo[k+1].ap));
      }
      cartAlpha.push_back(trainData.size()*1.0);          
      outfile<<"sqrt(alpha[k][k+1]) = ";             /*print alpha values*/
      for (size_t k=0 ; k<cartAlpha.size() ; k++) 
        outfile<<"("<<cartAlpha[k]<<")";                
    }
    else {  
      outfile<<"alpha_v[1..k] = ";   /*find the corresponding tree in CV data for each alpha[k][k+1]*/
      for (size_t k=0 ; k<cartAlpha.size() ; k++){
        bool found = false;    
        for (int n=treeInfo.size()-1 ; n>=0&&!found ; n--)
          if (cartAlpha[k] >= treeInfo[n].ap){             
            outfile<<"("<<treeInfo[n].ap<<")";  // find T^(CV)_v with alpha^(CV)_v [k] < alpha[k][k+1] 
            (cartTreeInfo[k].er).push_back(treeInfo[n].mi);  // add R^(CV) of tree T^(CV)_v at each stage
            found = true;           
          } 
        if (!found){
          outfile<<"("<<treeInfo[0].ap<<"*)"; // if alpha^(CV)_v [k] < alpha[k][k+1] not satisfied, 
          (cartTreeInfo[k].er).push_back(treeInfo[0].mi);  // return alpha^(CV)_v [0] = 0
        }        
      }    
    }  
    outfile<<endl;                                   /************/   
    stage++;  // proceed to the next stage 
    
  } // end all stage (subset)

  vector <node_info> bestNodeInfo;
  bestNodeInfo = printCartFinal(cartNodeInfo, cartTreeInfo, trainData.size(), trainData, valData.size(), valData);
  List traingROC_List = assessAUC(bestNodeInfo,0);  // calculate the averaged AUC for the best pruned tree based on training data
  List validationROC_List;
  if (valData.size()>0)
    validationROC_List = assessAUC(bestNodeInfo,1);  // calculate the averaged AUC for the best pruned tree based on validation data
    
  outfile<<"                         ----    END_OF_FILE    ----"<<endl;
  outfile.close();
  List treeNodeInfo = getTreeList(bestNodeInfo);
  return List::create(Named("Training_Data_ROC") = traingROC_List,
                      Named("Validation_Data_ROC") = validationROC_List,
                      Named("Tree_Node_Info") = treeNodeInfo);
}
