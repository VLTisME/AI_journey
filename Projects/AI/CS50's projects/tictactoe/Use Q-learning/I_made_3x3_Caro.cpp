#include "header.h"
#include "func.cpp"

  map <int, state> last;

  double get_future_reward(){
         if (winner()) return 0.0;
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
       if (Q.find(cur) == Q.end()) Q[cur] = 0.0;
       Q[cur] = Q[cur] + alpha * (r + best_future_reward - Q[cur]);
  }

  ii move(){
     double mx = -10000.0;
     ii best_move = {-1, -1};
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

     int r = rand() % 100;
     if (r <= 70){
         int sz = (int)ac.size();
         int k = rand() % sz;
         best_move = ac[k];
     }
     if (best_move.first == -1) return ac[0];
     return best_move;
  }

  void start(){
       for (int i = 0; i < 3; ++i){
            for (int j = 0; j < 3; ++j){
                 board[i][j] = 0;
            }
       }

       // AI learns to play against itself  
       last.clear();
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
              else if (last.find(-p) != last.end()){
                  update_q_value(last[-p], 0);
              }
       }
  }

  signed main(){
         srand(time(0));
         loadBot("bot_brain.txt");
         // train the AI 200000 times:
         int pc = 0;
         for (int t = 1; t <= 40000; ++t){
              if (t % 400 == 0) cout << ++pc << "% training...\n";
              start();
         }
         saveBot("bot_brain.txt");
         return 0;
  }