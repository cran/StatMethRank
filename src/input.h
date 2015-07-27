//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// input.h
#ifndef INPUT_H
#define INPUT_H

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include "data.h"
#include "question.h"

class Input{
public:
  Input(){};  
  ~Input(){};

  // Initialization
  void init(vector <Data>& d, vector <Question>& q, vector <double>& a, const string& info_file, const string& data_file, const int& rank_size, const int& item_size){    
    putAlpha(rank_size, item_size);   
    readInfo(info_file); 
    readData(data_file); 
    d = dat;
    q = que;
    a = alp;
  }
  
  // Read the question and target info, 
  // (1) store the question's : (i) description, (ii) number of possible values
  //                 (iii) type (discrete/continuous), (iv) all possible values
  // (2) store the possible targets
  void readInfo(const string& filename){ 
    ifstream data;
    string s;
    data.open(filename.c_str());
    
    for (int i=0 ; i<3 ; i++)             // line 1,2 of info_file          
      data>>s;    
    data>>s;
    int target_size = atoi(s.c_str());    // target size
    for (int i=0 ; i<target_size ; i++){                
      data>>s;         
      tar.push_back(atoi(s.c_str())); 
    }  
    for (int i=0 ; i<6 ; i++)             // line 4,5 of info_file          
      data>>s;   
      
    // question and choice
    while (data >> s){                      
      vector <double> choice;  
      des.push_back(s.c_str());           // description  
      data>>s;                
      string type = s.c_str();            // input type
      data>>s;                          
      int split_size = atoi(s.c_str());   // number of split
      data>>s;                          
      int choice_size = atoi(s.c_str());  // number of choice (=2 if type="c")  
      for (int i=0 ; i<choice_size ; i++){           
        data>>s;                          
        choice.push_back(atof(s.c_str()));// choices (=min/max value if type="c")
      }
      que.push_back(Question(split_size, type, choice));
    }   

    data.close();
  }
  
  // Read the data line by line 
  void readData(const string& filename){ 
    ifstream data;
    string s;
    int que_size = que.size();
    data.open(filename.c_str());

    // row_size: que_size + 5(id + rank 1,2,3 + # of rank) 
    for (int i=0 ; i<que_size+5 ; i++)        // dummy       
      data>>s;

    while (data >> s){  
      vector <double> curr_input;
      vector <int> curr_target;
      int curr_id = atoi(s.c_str());          // data_id      
      for (int i=0 ; i<3 ; i++){              // target (t1,t2,t3)
        data>>s;
        curr_target.push_back(atoi(s.c_str()));
      }
      data>>s;                                // number of rank
      int curr_nor = atoi(s.c_str());                               
      for (int i=0 ; i<que_size ; i++){       // input (q0-q7)
        data>>s;
        curr_input.push_back(atof(s.c_str()));
      }       
      if (curr_nor > 0)                       // data is useful only if NOR > 0
        dat.push_back(Data(curr_input, curr_target, curr_id, curr_nor));     
    }
    data.close();
  }  

  // Store the alpha value for Foot Rule Accessment
  void putAlpha(const int& rank_size, const int& item_size){
    int k = rank_size;
    if (k==1 || k!=item_size)
      k++;
    int x=0;
    for (int i=1 ; i<=k ; i++)
      x+=i;
    for (int i=k ; i>=1 ; i--)
      alp.push_back(1.0*i/x);
  }
  
  
  // Get the description of the x-th question
  string getDescription(int& x)const{
    return des[x];
  }

  // Pre: input the value of the x-th question
  // Post: return the position of that option
  int find(const int& x, const double& option)const{
    vector <double> q = que[x].getChoice();
    for (size_t i=0 ; i<q.size() ; i++)
      if (q[i] == option)
        return i;
    return -1;
  }
  
  // Return the x-th choice of question q
  double findInput(const int& q, const int& x)const{
    return que[q].getChoice(x);
  }
  
  // Pre: input the value of the target
  // Post: return the position of that option 
  int find(const int& option)const{
    for (size_t i=0 ; i<tar.size() ; i++)
      if (tar[i] == option)
        return i;
    return -1;
  }
  
  // Return the x-th choice of the target
  int findTarget(const int& x)const{
    return tar[x];
  }

private:
  vector <Question> que;
  vector <Data> dat;
  vector <int> tar;
  vector <string> des;
  vector <double> alp;
  
};
// end of class Input
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#endif
