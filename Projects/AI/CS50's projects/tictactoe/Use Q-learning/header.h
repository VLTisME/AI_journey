#ifndef _HEADER_H_
#define _HEADER_H_

#include <iostream>
#include <map>
#include <vector>
#include <tuple>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <fstream>


using namespace std;

  typedef pair <int, int> ii;

  const double alpha = 0.92;
  const double theta = 0.8;

  struct state{
         vector <vector<int>> board;
         ii action;
         
         bool operator<(const state& other) const {
         return tie(board, action) < tie(other.board, other.action);
    }
  };


  map <state, double> Q;
  vector<vector<int>> board(3, vector<int>(3, 0));

  int winner();
  int player();
  void loadBot(const string& filename);
  void saveBot(const string& filename);
  void display();

  #endif