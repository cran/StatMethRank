//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// data.h and data.cpp
#include "data.h"

Data::Data(const vector<double>& i, const vector<int>& t, const int& id_, const int& nor_)
  : input(i), target(t), id(id_), nor(nor_){}

Data::~Data(){}

double Data::getInput(const int& x) const{
  return input[x];
}

int Data::getTarget(const int& x) const{
  return target[x];
}

int Data::getItemRank(const int& x) const{
  for (size_t i=0 ; i<target.size() ; i++)
    if (target[i] == x)
      return (i+1);
  return -1;    
}

int Data::getID() const{
  return id;
}

int Data::getNOR() const{
  return nor;
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
