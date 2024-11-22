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

  const double alpha = 0.9;
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
  map <int, state> last;
  
  int winner(){
      for (int i = 0; i < 3; i++) {
           if (board[i][0] != 0 && board[i][0] == board[i][1] && board[i][1] == board[i][2]) 
               return board[i][0];
           if (board[0][i] != 0 && board[0][i] == board[1][i] && board[1][i] == board[2][i]) 
               return board[0][i];
      }
     
      if (board[0][0] != 0 && board[0][0] == board[1][1] && board[1][1] == board[2][2]) 
          return board[0][0];
      if (board[0][2] != 0 && board[0][2] == board[1][1] && board[1][1] == board[2][0]) 
          return board[0][2];
      return 0;
  }

  int player(){
       int x = 0, o = 0;
       for (int i = 0; i < 3; i++){
           for (int j = 0; j < 3; j++){
                if (board[i][j] == 1) x++;
                if (board[i][j] == -1) o++;
           }
       }
       return x == o ? 1 : -1;
  }

  ii move(){
     double mx = -10000.0;
     ii best_move;
     vector <ii> ac;
     for (int i = 0; i < 3; ++i){
          for (int j = 0; j < 3; ++j){
               if (board[i][j] == 0){
                   ac.push_back({i, j});
                   if (Q.find({board, {i, j}}) != Q.end() && Q[{board, {i, j}}] >= mx){
                       mx = Q[{board, {i, j}}];
                       best_move = {i, j};
                   }
               }
          }
     }
     if (!ac.size()) return {-1, -1};

     int r = rand() % 10;
     if (r <= 1){
         int sz = (int)ac.size();
         int k = rand() % sz;
         return ac[k];
     }
     return best_move;
  }

  void display(){
       system("cls");
       for (int i = 0; i < 3; ++i){
            for (int j = 0; j < 3; ++j){
                 if (board[i][j] == 0) cout << "_ ";
                 else if (board[i][j] == 1) cout << "X ";
                 else cout << "O ";
            }
            cout << "\n";
       }
  }

  double get_future_reward(){
         double mx = -1e9; 
         bool has_action = false;
         for (int i = 0; i < 3; i++){
             for (int j = 0; j < 3; j++){
                  if (board[i][j] == 0 && Q.find({board, {i, j}}) != Q.end()) {
                      has_action = true;
                      mx = max(mx, Q[{board, {i, j}}]);
                  }
             }
         }
         return has_action ? mx : 0.0; 
  }

  void update_q_value(state cur, int r){
       int best_future_reward = get_future_reward();
       Q[cur] = Q[cur] + alpha * (r + theta * best_future_reward - Q[cur]);
  }

  void start(){
       for (int i = 0; i < 3; ++i){
            for (int j = 0; j < 3; ++j){
                 board[i][j] = 0;
            }
       }

       // AI learns to play against itself  
       int dem = 0;
       while (true){
              ii action = move();
              if (action.first == -1) break;
              state cur = {board, action};
              int p = player();
              board[action.first][action.second] = p;
              last[p] = cur;

              if (winner()){
                  update_q_value(cur, 1);
                  update_q_value(last[-p], -1);
                  break;
              }
              else if (dem > 0){
                  update_q_value(last[-p], 0);
              }
              ++dem;
       }
  }

  signed main(){
         srand(time(0));
         //loadBot("bot_brain.txt");
         // train the AI 100000 times:
         for (int t = 0; t < 30000; ++t){
              start();
         }
         
         
         for (int i = 0; i < 3; ++i){
              for (int j = 0; j < 3; ++j){
                   board[i][j] = 0;
              }
         }

         while (true){
                int p = player();
                display();
                if (winner() != 0){
                    cout << (winner() == 1 ? "You win! " : "AI wins!");
                    break;
                }
                if (move().first == -1){
                    cout << "Draw!";
                    break;
                }
                if (p == 1){
                    // your turn 
                    int x, y;
                    while (1){
                           cout << "Enter your move: ";
                           cin >> x >> y;
                           --x, --y;
                           if (x < 0 || y < 0 || x > 2 || y > 2 || board[x][y] != 0){
                               cout << "Invalid move. Try again!\n"; 
                           }
                           if (board[x][y] == 0){
                               board[x][y] = 1;
                               break;
                           }
                    }
                }
                else{
                    ii action = move();
                    board[action.first][action.second] = -1;
                }
         }
         system("pause");
         return 0;
  }