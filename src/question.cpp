// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// question.h and question.cpp
#include "question.h"

Question::Question(const int& i, const string& t, const vector<double>& v)
  : num_choice(i), type(t), choice(v){}

Question::~Question(){}

int Question::getSize()const{
  return num_choice;
}

string Question::getType()const{
  return type;
}

vector<double> Question::getChoice()const{
  return choice;
}

double Question::getChoice(const int& x)const{
  return choice[x];
}
// end of class Question
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
