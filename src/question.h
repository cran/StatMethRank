#ifndef QUESTION_H
#define QUESTION_H

#include <vector>
#include <string>

using namespace std;

class Question{
public:
  Question(const int&, const string&, const vector<double>&);
  ~Question();
  int getSize()const;                // return the number of values during split
  string getType()const;             // return the type of option
  vector<double> getChoice()const;   // return the whole vector of choice
  double getChoice(const int&)const; // return the x-th choice of the question
private:
  int num_choice;
  string type;
  vector <double> choice;
};

#endif
