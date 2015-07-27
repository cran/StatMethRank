#ifndef UTILITY_H
#define UTILITY_H

#include <vector>

struct pair_comp{
  int item1;
  int item2;
  double exp_p;
  double obs_p;
  pair_comp(int i, int j, double e, double o) :
    item1(i), item2(j), exp_p(e), obs_p(o) {}
};

struct node_info{
  vector <Data> mbr;  // node members (training set)
  vector <Data> smbr; // node members (test set)
  string id;          // node id
  double rv;          // r_value
  double gv;          // g_value (for leave node, g_value = 0)
  int fn;             // final node (1: final node; 0: internal node, -1: pruned node)
  int st;             // split type (0: continuous; 1: discrete)
  int sv0;            // split variable
  double sv1;         // split value (unuse for split type 1)
  vector <int> sv2;   // split value (unuse for split type 0)
  node_info(vector <Data> d, vector <Data> s, string i, double r, double g, 
            int f, int t, int v0, double v1, vector <int> v2) : mbr(d), smbr(s), 
            id(i), rv(r), gv(g), fn(f), st(t), sv0(v0), sv1(v1), sv2(v2) {}
};

struct tree_info{
  int id;             // tree id
  double ap;          // alpha value
  int size;           // number of final nodes
  double mi;          // error value
  vector <double> er; // accumulate error value (for the final tree)
  tree_info(int i, double a, int s, double m, vector <double> e) 
           : id(i), ap(a), size(s), mi(m), er(e){}   
};

struct AUCpair{ //vector for storing frequency of each item pair for calculating AUC
  int item1;  // item 1
  int item2;  // item 2 
  string id;  // node id
  int nsize;  // node size
  double pwprob; //pairwise probability
  AUCpair(int i, int j, string m, int k, double p): item1(i), item2(j), id(m), nsize(k), pwprob(p){}
};

//typedef struct AUC{ // vector for storing node size and pairwise frequency (all pairs) for each node
//  int nodesize;     // save for calculation of AUC
//  vector <AUCpair> pairs;
//  AUC(int n, vector <AUCpair> d): nodesize(n), pairs(d) {}
//};

class Utility{
public:
  Utility(){}
  ~Utility(){}

  // Post: generate all possible size-k subset of a set (with duplicate)
  void k_combin(int m, int n, int k, int r[], vector <vector <int> > & vec) {
    int i;
    if (m <= k){ 
      for (r[m]=r[m-1]+1 ; r[m]<=n ; r[m]++)
        k_combin(m+1, n, k, r, vec);
    } 
    else {
      vector <int> v;
      for (i=1; i<=k; i++)
        v.push_back(r[i]-1);    
      vec.push_back(v);
    }
  }

  // Post: generate all possible size-k subset of a set (with duplicate)
  void k_combin(int m, int n, int k, vector<int> & r, vector <vector <int> > & vec) {
      int i;
	  if (m <= k){
		  for (r[m] = r[m - 1] + 1; r[m] <= n; r[m]++)
			  k_combin(m + 1, n, k, r, vec);
	  }
	  else {
		  vector <int> v;
		  for (i = 1; i <= k; i++)
			  v.push_back(r[i] - 1);
		  vec.push_back(v);
	  }
  }

  // Post: generate all possible subset of a set 
  vector <vector <int> > gen_com(int n)
  {
    vector <vector <int> > vec;		// store all subset of size-n set
    
    // int r[n /2];
  //int* r = new int[n/2];
  int r[255];
  r[0] = 0;
  
	/*
	vector<int> r(n / 2); //all elements are initialized to 0
	r[0] = 0;
	*/
    for (int k=1 ; k<=n/2 ; k++)
      k_combin(1, n, k, r, vec);
  
    if (n%2 == 0){			// remove duplicate set
      int tmp = 1;
      for (int i=n ; i>n/2 ; i--)
        tmp *= i;
      for (int i=1 ; i<=n/2 ; i++)
        tmp /= i;
      tmp /= 2;
      for (int i=0 ; i<tmp ; i++)
        vec.pop_back();
    }
    
    return vec;
  }
/*
  // Post: store all valid ranking {x1,x2,i} to "all_perm"
  void perm3(vector <vector <int> >& all_perm, int size, int x1, int x2){
    for (int i=1 ; i<=size ; i++)
      if (i!=x1 && i!=x2){
        vector <int> v1;
        v1.push_back(x1);
        v1.push_back(x2);
        v1.push_back(i);
        all_perm.push_back(v1);
      }
  }
  
  // call by perm1 and call perm3 recursively
  void perm2(vector <vector <int> >& all_perm, int size, int x1){
    for (int i=1 ; i<=size ; i++)
      if (i != x1)
        perm3(all_perm, size, x1, i);
  }
  
  // Post: generate all possible permutation of ranking top 3 from x obj
  vector<vector <int> > perm1(int size){
    vector <vector <int> > all_perm;
    for (int i=1 ; i<=size ; i++)
      perm2(all_perm, size, i);
    return all_perm;
  }
*/

  void perm4(vector <vector <double> >& all_perm, vector <double> v0, int x1, int x2, int x3){
    if (v0.size() == 4){
      for (int i=0 ; i<v0.size() ; i++)      
        if (i!=x1 && i!=x2 && i!=x3){
          vector <double> vx;    
          vx.push_back(v0[x1]);
          vx.push_back(v0[x2]);
          vx.push_back(v0[x3]);
          vx.push_back(v0[i]);
          all_perm.push_back(vx);
        }  
    }     
  }

  void perm3(vector <vector <double> >& all_perm, vector <double> v0, int x1, int x2){
    if (v0.size() == 3){
      for (int i=0 ; i<v0.size() ; i++)      
        if (i!=x1 && i!=x2){
          vector <double> vx;    
          vx.push_back(v0[x1]);
          vx.push_back(v0[x2]);
          vx.push_back(v0[i]);
          all_perm.push_back(vx);
        }  
    }
    else {     
      for (int i=0 ; i<v0.size() ; i++)
        if (i!=x1 && i!=x2)
          perm4(all_perm, v0, x1, x2, i);
    }      
  }
  
  void perm2(vector <vector <double> >& all_perm, vector <double> v0, int x1){
    if (v0.size() == 2){
      for (int i=0 ; i<v0.size() ; i++)      
        if (i != x1){
          vector <double> vx;    
          vx.push_back(v0[x1]);
          vx.push_back(v0[i]);
          all_perm.push_back(vx);
        }  
    }
    else {     
      for (int i=0 ; i<v0.size() ; i++)
        if (i != x1)
          perm3(all_perm, v0, x1, i);
    }
  }
  
  vector<vector <double> > perm1(vector <double> v0){
    vector <vector <double> > all_perm;
    for (int i=0 ; i<v0.size() ; i++){
      perm2(all_perm, v0, i);
    }            
    return all_perm;
  }

  // Post: return the maxmium obtainable discrepancy
  double maxDiscrepancy(vector<double> alpha, vector<double> v0){
    vector <double> r_star;
    double discrepancy_star = 0.0;    
    vector<vector <double> > all_perm = perm1(v0);    
    for (int i=0 ; i<all_perm.size() ; i++){
      double discrepancy = 0.0;        
      vector <double> vy = all_perm[i];
      for (int j=0 ; j<vy.size() ; j++) 
        discrepancy += alpha[j]*fabs(v0[j]-vy[j]);
      if (discrepancy > discrepancy_star){
        discrepancy_star = discrepancy;
        r_star = all_perm[i];
      }
    }         
    return discrepancy_star;
  }   

  // Post: return the mean
  double mean(vector <double> data){
    double tmp = 0.0;
    for (size_t i=0 ; i<data.size() ; i++)
      tmp += data[i];
    return (tmp/data.size());    
  }

  // Post: return the Standard Error (Sum[(Di-mean)^2] / N)
  double standardError(vector <double> data, double mean){
    double sum = 0.0;
    for (size_t i=0 ; i<data.size() ; i++)
      sum += (data[i]-mean)*(data[i]-mean);
    return (sum/data.size());    
  }
  
  // Multiply a vector with its transpose, then add to a base matrix
  vector <vector <double> > vecToMatrix(const vector <double>& v, const vector <vector <double> >& matrix){
    vector <vector <double> > fmatrix = matrix;
    for (size_t i=0 ; i<v.size() ; i++)
      for (size_t j=0 ; j<v.size() ; j++)
        fmatrix[i][j] = fmatrix[i][j] + 1.0*v[i]*v[j];
    return fmatrix;    
  }
  
  // Multiply a matrix with a magnitude
  vector <vector <double> > multconstMatrix(double m, const vector <vector <double> >& matrix){
    vector <vector <double> > fmatrix = matrix;
    for (size_t i=0 ; i<fmatrix.size() ; i++)
      for (size_t j=0 ; j<fmatrix[i].size() ; j++)
        fmatrix[i][j] = fmatrix[i][j]*m;
    return fmatrix;
  }
  
  // Multiply a (n*n) matrix with a (n*1) matrix (column vector)
  vector <double> multColMatrix(const vector <vector <double> >& matrix, const vector <double>& cmatrix){
    vector <double> fmatrix;
    for (size_t i=0 ; i<matrix.size() ; i++){
      double total = 0.0;
      for (size_t j=0 ; j<matrix[i].size() ; j++)
        total += matrix[i][j]*cmatrix[j];
      fmatrix.push_back(total);
    }  
    return fmatrix;
  }

  // Multiply a (1*n) matrix (row vector) with a (n*n) matrix
  vector <double> multRowMatrix(const vector <vector <double> >& matrix, const vector <double>& rmatrix){
    vector <double> fmatrix;
    for (size_t i=0 ; i<matrix.size() ; i++){
      double total = 0.0;
      for (size_t j=0 ; j<matrix[i].size() ; j++)
        total += matrix[j][i]*rmatrix[j];
      fmatrix.push_back(total);
    }  
    return fmatrix;    
  }  
  
  // Multiply a (1*n) matrix (row vector) with a (n*1) matrix (column vector)
  double multRowCol(const vector <double> & rmatrix, const vector <double>& cmatrix){
    double fmatrix = 0.0;
    for (size_t i=0 ; i<rmatrix.size() ; i++)
      fmatrix += rmatrix[i]*cmatrix[i];
    return fmatrix;  
  } 
  
  // Multiply a (m*n) matrix with a (n*q) matrix
  vector <vector <double> > multMatrix(const vector <vector <double> >& amat, const vector <vector <double> >& bmat){
    vector <double> vec1;
    vector <vector <double> > fmat;
    for (size_t i=0 ; i<bmat[0].size(); i++)
      vec1.push_back(0.0);
     
    for (size_t i=0 ; i<amat.size(); i++)
      fmat.push_back(vec1);
    for (size_t i=0 ; i<amat.size(); i++){
      for (size_t j=0 ; j<bmat[0].size(); j++)
        for (size_t k=0 ; k<bmat.size(); k++)
          fmat[i][j] += amat[i][k]*bmat[k][j]*1.0;
    }  
    return fmat;
  }  

  // Multiply a matrix with a magnitude
  vector <vector <double> > multMatrix(double m, const vector <vector <double> >& matrix){
    vector <vector <double> > fmatrix = matrix;
    for (size_t i=0 ; i<fmatrix.size() ; i++)
      for (size_t j=0 ; j<fmatrix[i].size() ; j++)
        fmatrix[i][j] = fmatrix[i][j]*m;
    return fmatrix;
  }


  // Multiply a transpose matrix with itself
  vector <vector <double> > multMat_MatTp(const vector <vector <double> >& mat){
    vector <double> vec1, vec2;
    vector <vector <double> > mat_t, fmat;
//    for (size_t i=0 ; i<mat[0].size(); i++)
//      vec2.push_back(0.0);
//    for (size_t i=0 ; i<mat.size(); i++)
//      mat_t.push_back(vec2);
//    for (size_t i=0 ; i<mat.size(); i++)
//      for (size_t j=0 ; j<mat[0].size(); j++)
//        mat_t[j][i] = mat[i][j];
    
    for (size_t i=0 ; i<mat[0].size(); i++)
      vec1.push_back(0.0);
    for (size_t i=0 ; i<mat[0].size(); i++)
      fmat.push_back(vec1);

    for (size_t i=0 ; i<mat[0].size(); i++){
      for (size_t j=0 ; j<mat[0].size(); j++)
        for (size_t k=0 ; k<mat.size(); k++)
          fmat[i][j] += mat[k][i]*mat[k][j]*1.0;
    }  
    return fmat; 
  }
  
  
  // LU decomposition of Matrix
  vector <vector <double> > luDecomp(const vector <vector <double> >& amat){

    int n = amat.size();
    double temp = 0.0;
    double big = 0.0;
    double sum = 0.0;
    vector <vector <double> > ludarray = amat;
    vector <double> vv;
   
    for (int i=0 ; i<n ; i++){
      big = 0.0;
      for (int j=0 ; j<n ; j++)
        if  ((temp=fabs(ludarray[i][j])) > big) 
          big = temp;
      if (big == 0.0){
        // cout<<"Attempted LU Decomposition of a singular matrix in Matrix.luDecomp()";
        return ludarray;
      }    
      vv.push_back(1.0/big);
    }
  
    for (int j=0 ; j<n ; j++) {
      for (int i=0 ; i<j ; i++) {
        sum = ludarray[i][j];
        for (int k=0 ; k<i ; k++) 
          sum -= ludarray[i][k]*ludarray[k][j];
        ludarray[i][j] = sum;
      }
      for (int i=j ; i<n ; i++) {
        sum=ludarray[i][j];
        for (int k=0 ; k<j ; k++)
          sum -= ludarray[i][k]*ludarray[k][j];
        ludarray[i][j] = sum;      
      }
 
      if(ludarray[j][j] == 0.0)
        ludarray[j][j] = 0.0;  //TINY
      if(j != n-1) {
        temp = 1.0/ludarray[j][j];
        for (int i=j+1 ; i<n ; i++)
          ludarray[i][j] *= temp;    	
      }
    }
  
    return ludarray;
  }

  // Back substitution
  vector <vector <double> > backSub(const vector <vector <double> >& lmat){

    vector <vector <double> > vv1 = lmat;
    for (size_t i=0 ; i<lmat.size() ; i++){
      vector <double> v1, x1, z1;
      for (int j=0 ; j<lmat.size() ; j++){
        z1.push_back(0.0);
        x1.push_back(0.0);
        if (j == i)
          v1.push_back(1.0);
        else v1.push_back(0.0);
      }
      for (int j=0 ; j<lmat.size() ; j++){
        double sum = v1[j];
        for (int k=0 ; k<j ; k++)    
          sum -= lmat[j][k]*z1[k];  
        z1[j] = sum; 
      }
 
      for (int j=lmat.size()-1 ; j>=0 ; j--){
        double sum = z1[j];
        for (int k=lmat.size()-1 ; k>j ; k--)    
          sum -= lmat[j][k]*x1[k];  
        x1[j] = sum/lmat[j][j];
      }
      for (int j=0 ; j<lmat.size() ; j++)
        vv1[j][i] = x1[j];
    }

    return vv1;
  }

/*
  vector <vector <double> > matInverse(vector <vector <double> > amat){
 
    amat[0][0] = 25.0;  amat[0][1] = 5.0;   amat[0][2] = 1.0;
    amat[1][0] = 64.0;  amat[1][1] = 8.0;   amat[1][2] = 1.0;
    amat[2][0] = 144.0; amat[2][1] = 12.0;  amat[2][2] = 1.0;
    cout<<"amat:"<<endl;
    for (size_t i=0 ; i<amat.size() ; i++){
      for (size_t j=0 ; j<amat.size() ; j++)
        cout<<amat[i][j]<<", ";
      cout<<endl;
    }	
  
    vector <vector <double> > ludmat = luDecomp(amat);  
    cout<<"ludmat:"<<endl;
    for (size_t i=0 ; i<ludmat.size() ; i++){
      for (size_t j=0 ; j<ludmat.size() ; j++)
        cout<<ludmat[i][j]<<", ";
      cout<<endl;
    }	    	

    vector <vector <double> > imat = backSub(ludmat);
    cout<<"imat:"<<endl;
    for (size_t i=0 ; i<imat.size() ; i++){
      for (size_t j=0 ; j<imat.size() ; j++)
        cout<<imat[i][j]<<", ";
      cout<<endl;
    }	
 
    return imat;
  }
*/  

  // get the value from Chi-Square table
  double chiSqrValue(double alp, int deg){
    
    double alpha[] = {0.995,     0.99,     0.975,    0.95,       0.2,     0.1,    0.05,   0.025,    0.01,   0.005};
    double val01[] = {0.0000393, 0.000157, 0.000982, 0.00393,  1.642,   2.706,   3.841,   5.024,   6.635,   7.879};
    double val02[] = {0.1000,    0.0201,   0.0506,   0.103,    3.219,   4.605,   5.991,   7.378,   9.210,  10.597};
    double val03[] = {0.0717,    0.115,    0.216,    0.352,    4.642,   6.251,   7.815,   9.348,  11.345,  12.838};
    double val04[] = {0.207,     0.297,    0.484,    0.711,    5.989,   7.779,   9.488,  11.143,  13.277,  14.860};
    double val05[] = {0.412,     0.554,    0.831,    1.145,    7.289,   9.236,  11.070,  12.832,  15.086,  16.750};
    double val06[] = {0.676,     0.872,    1.237,    1.635,    8.558,  10.645,  12.592,  14.449,  16.812,  18.548};
    double val07[] = {0.989,     1.239,    1.690,    2.167,    9.803,  12.017,  14.067,  16.013,  18.475,  20.278};
    double val08[] = {1.344,     1.646,    2.180,    2.733,   11.030,  13.362,  15.507,  17.535,  20.090,  21.955};
    double val09[] = {1.735,     2.088,    2.700,    3.325,   12.242,  14.684,  16.919,  19.023,  21.666,  23.589};
    double val10[] = {2.156,     2.558,    3.247,    3.940,   13.442,  15.987,  18.307,  20.483,  23.209,  25.188};
    double val11[] = {2.603,     3.053,    3.816,    4.575,   14.631,  17.275,  19.675,  21.920,  24.725,  26.757};
    double val12[] = {3.074,     3.571,    4.404,    5.226,   15.812,  18.549,  21.026,  23.337,  26.217,  28.300};
    double val18[] = {6.265,     7.015,    8.231,    9.390,   22.760,  25.989,  28.869,  31.526,  34.805,  37.156};
    double val19[] = {6.844,     7.633,    8.907,    10.117,  23.900,  27.204,  30.144,  32.852,  36.191,  38.582};
    double val20[] = {7.434,     8.260,    9.591,   10.851,   25.038,  28.412,  31.410,  34.170,  37.566,  39.997};
    double val21[] = {8.034,     8.897,   10.283,   11.591,   26.171,  29.615,  32.671,  35.479,  38.932,  41.401};
    double val23[] = {9.260,    10.196,   11.689,   13.091,   28.429,  32.007,  35.172,  38.076,  41.638,  44.181};
    double val24[] = {9.886,    10.856,   12.401,   13.848,   29.553,  33.196,  36.415,  39.364,  42.980,  45.559};
    double val30[] = {13.787,   14.953,   16.791,   18.493,   36.250,  40.256,  43.773,  46.979,  50.892,  53.672};
    double val42[] = {22.138,   23.650,   25.999,   28.144,   49.456,  54.090,  58.124,  61.777,  66.206,  69.336};
    double val56[] = {32.490,   34.350,   37.212,   39.801,   64.658,  69.919,  74.468,  78.567,  83.513,  86.994};
    double val60[] = {35.534,   37.485,   40.482,   43.188,   68.972,  74.397,  79.082,  83.298,  88.379,  91.952};
    double val72[] = {44.843,   47.051,   50.428,   53.462,   81.857,  87.743,  92.808,  97.353, 102.816, 106.648};
    double val90[] = {59.196,   61.754,   65.647,   69.126,  101.054, 107.565, 113.145, 118.136, 124.116, 128.299};
    double val120[]= {83.852,   86.923,   91.573,   95.705,  132.806, 140.233, 146.567, 152.211, 158.950, 163.648};
    double val210[]= {160.969, 165.283,  171.759,  177.465,  227.025, 236.655, 244.808, 252.027, 260.595, 266.537};
    double val336[]= {272.985, 278.650,  287.112,  294.527,  357.600, 369.620, 379.746, 388.675, 399.230, 406.523};
    double val504[]= {425.978, 433.094,  443.688,  452.939,  530.508, 545.092, 557.335, 568.099, 580.787, 589.532};
    double val720[]= {626.012, 634.673,  647.537,  658.740,  751.727, 769.040, 783.534, 796.251, 811.208, 821.499};

    int x = -1;    
    for (int i=0 ; i<10 ; i++)
      if (alpha[i] == alp){
        x = i;
        break;
      }
    /* Error */    
    /*
    if (deg<1 || deg>720 || x==-1){
      cout<<"Error: Chi-Square Value out of range"<<endl;
      exit(0);          
    }
    */
       
    if (deg == 1)       return val01[x];
    else if (deg == 2)  return val02[x];
    else if (deg == 3)  return val03[x]; 
    else if (deg == 4)  return val04[x];   
    else if (deg == 5)  return val05[x];
    else if (deg == 6)  return val06[x];
    else if (deg == 7)  return val07[x];
    else if (deg == 8)  return val08[x];   
    else if (deg == 9)  return val09[x];
    else if (deg == 10) return val10[x];
    else if (deg == 11) return val11[x];
    else if (deg == 12) return val12[x];
    else if (deg == 18) return val19[x];
    else if (deg == 19) return val19[x];
    else if (deg == 20) return val20[x];
    else if (deg == 21) return val21[x];
    else if (deg == 23) return val23[x];
    else if (deg == 24) return val24[x];
    else if (deg == 30) return val30[x];
    else if (deg == 42) return val42[x];
    else if (deg == 56) return val56[x];
    else if (deg == 60) return val60[x];
    else if (deg == 72) return val72[x];
    else if (deg == 90) return val90[x];
    else if (deg == 120) return val120[x];
    else if (deg == 210) return val210[x];
    else if (deg == 336) return val336[x];
    else if (deg == 504) return val504[x];
    else if (deg == 720) return val720[x];
    else return -1;
    }  





  /*
  // Post: store all valid ranking {x1,x2,i} to "all_perm"
  void perm3(vector <vector <int> >& all_perm, int size, int x1, int x2){
    for (int i=1 ; i<=size ; i++)
      if (i!=x1 && i!=x2){
        vector <int> v1;
        v1.push_back(x1);
        v1.push_back(x2);
        v1.push_back(i);
        all_perm.push_back(v1);
      }
  }
  
  // call by perm1 and call perm3 recursively
  void perm2(vector <vector <int> >& all_perm, int size, int x1){
    for (int i=1 ; i<=size ; i++)
      if (i != x1)
        perm3(all_perm, size, x1, i);
  }
  
  // Post: generate all possible permutation of ranking top 3 from x obj
  vector<vector <int> > perm1(int size){
    vector <vector <int> > all_perm;
    for (int i=1 ; i<=size ; i++)
      perm2(all_perm, size, i);
    return all_perm;
  }
  */
 

  /*
  vector <vector <double> > matInverse(vector <vector <double> > amat){
 
    amat[0][0] = 25.0;  amat[0][1] = 5.0;   amat[0][2] = 1.0;
    amat[1][0] = 64.0;  amat[1][1] = 8.0;   amat[1][2] = 1.0;
    amat[2][0] = 144.0; amat[2][1] = 12.0;  amat[2][2] = 1.0;
    cout<<"amat:"<<endl;
    for (size_t i=0 ; i<amat.size() ; i++){
      for (size_t j=0 ; j<amat.size() ; j++)
        cout<<amat[i][j]<<", ";
      cout<<endl;
    } 
  
    vector <vector <double> > ludmat = luDecomp(amat);  
    cout<<"ludmat:"<<endl;
    for (size_t i=0 ; i<ludmat.size() ; i++){
      for (size_t j=0 ; j<ludmat.size() ; j++)
        cout<<ludmat[i][j]<<", ";
      cout<<endl;
    }       

    vector <vector <double> > imat = backSub(ludmat);
    cout<<"imat:"<<endl;
    for (size_t i=0 ; i<imat.size() ; i++){
      for (size_t j=0 ; j<imat.size() ; j++)
        cout<<imat[i][j]<<", ";
      cout<<endl;
    } 
 
    return imat;
  }
  */ 
};
// end of class Utility
#endif
