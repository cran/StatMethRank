#ifndef DATA_H
#define DATA_H

#include <vector>
#include <string>

using namespace std;

class Data{
public:
  Data(const vector <double>&, const vector <int>&, const int&, const int&);
  ~Data();
  double getInput(const int&)const;  // return the x-th input value
  int getTarget(const int&)const;    // return the x-th target value
  int getItemRank(const int&)const;  // return the rank of item-x
  int getID()const;                  // return the data id
  int getNOR()const;                 // return the number of rank
private:
  vector <double> input;
  vector <int> target;
  int id;
  int nor;
};

#endif
