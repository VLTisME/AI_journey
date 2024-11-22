#include "header.h"

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

  void loadBot(const string& filename) {
    ifstream inFile(filename);
    if (!inFile) {
        cerr << "Error: Unable to open file " << filename << " for reading!\n";
        return;
    }

    cout << "Loading Q-learning data from " << filename << "...\n";

    while (true) {
        state s;
        double val;
        s.board.resize(3, vector<int>(3));

        for (auto &row: s.board){
            for (int &x: row){
                if (!(inFile >> x)){
                    inFile.close();
                    return;
                }
            }
        }
        if (!(inFile >> s.action.first >> s.action.second)) {
            inFile.close();
            return;
        }
        if (!(inFile >> val)) {
            inFile.close();
            return;
        }
        Q[s] = val;
    }

    cout << "Done loading!\n";
}

  void saveBot(const string& filename){
       ofstream outFile(filename);
       if (!outFile){
           cerr << "Error: Unable to open file for saving!\n";
           return;
       }

       for (auto &entry: Q){
            state s = entry.first;
            double val = entry.second;
            for (auto &row: s.board){
                 for (int &x: row){
                      outFile << x << " ";
                 }
            }
            outFile << s.action.first << " " << s.action.second << " ";
            outFile << val << '\n';
       }

       outFile.close();

       cout << "Done saving!\n";
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
