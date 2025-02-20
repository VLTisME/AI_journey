#include <cstdlib>
#include <thread>
#include "header.h"
#include "func.cpp"

void playMusic() {
    string command = "powershell -c (New-Object Media.SoundPlayer 'In-the-hall-of-the-mountain-king.wav').PlaySync();";
    system(command.c_str());
}

  signed main(){

         thread musicThread(playMusic);

         srand(time(0));
         // con 2 nhiem vu:
         // 1: am thanh
         // 2: hinh anh GUI
         // 3: toi uu lai code. Train ~ 30k lan thoi. Tham chi cho du minh co train 5e5 lan thi no van danh ngu --> sai logic
         // 6: CS50 && Kaggle
         // 7: chuan bi bai kiem diem cong ngay mai
         // 8: push len github
         // 9: đặt vé tàu - CS50, kaggle, caro, gomoku, ai,… chỉ tuấn sound, gửi tictactoe Q learning cho cao, học toán RR & linear algebra
         // 10: ngay mai co deadline cslt gi vay?


         loadBot("bot_brain.txt");

         for (int i = 0; i < 3; ++i){
              for (int j = 0; j < 3; ++j){
                   board[i][j] = 0;
              }
         }
         while (true){
                while (true){
                       int p = player();
                       display();
                       if (winner() != 0){
                       cout << (winner() == 1 ? "You win! " : "AI wins!");
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
                                  else if (board[x][y] == 0){
                                      board[x][y] = 1;
                                      break;
                                  }
                           }
                       }
                       else{
                           double mx = -10000.0;
                           ii action = {-1, -1};
                           for (int i = 0; i < 3; ++i){
                               for (int j = 0; j < 3; ++j){
                                   if (board[i][j] == 0){
                                      if (Q.find({board, {i, j}}) != Q.end() && Q[{board, {i, j}}] > mx){
                                         mx = Q[{board, {i, j}}];
                                         action = {i, j};
                                      }
                                   }
                               }
                           }
                           if (action.first == -1){
                              cout << "Draw!\n";
                              break;
                           }
                           board[action.first][action.second] = -1;
                       }
                }
                cout << "\nPress 1 to play again, 0 to exit: ";
                int x;
                cin >> x;
                if (x == 0) break;
                for (int i = 0; i < 3; ++i){
                     for (int j = 0; j < 3; ++j){
                          board[i][j] = 0;
                     }
                }
         }
         system("pause");
         return 0;
         this_thread::sleep_for(std::chrono::seconds(10));
         musicThread.join();
  }