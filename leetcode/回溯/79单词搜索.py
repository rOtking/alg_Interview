class Solution:
    def __init__(self):
            self.directions = [(1,0), (-1,0), (1,0), (-1,0)]
    def exist(self, board, word: str) -> bool:
        return self.dfs(track=[(0, 0)], board=board, x=0, y=0, word=word)

    def dfs(self, track, board, x, y, word):
        '''
        track是走过的坐标
        xy  是当前位置的坐标
        '''

        combine_word = ''
        for cor in track:
            combine_word += board[cor[0]][cor[1]]

        if combine_word == word:
            return True

        for di in self.directions:
            # 走过的不重复
            cor_x = x + di[0]
            cor_y = y + di[1]
            if (cor_x, cor_y) in track:
                continue
            if cor_x >= len(board) or cor_x < 0 or cor_y >= len(board[0]) or cor_y < 0:
                break

            track.append((cor_x, cor_y))
            flag = self.dfs(track, board, cor_x, cor_y, word)
            track.pop()
            if flag:
                return flag

        return False


if __name__ == '__main__':
    s = Solution()
    res = s.exist(board=[["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word="ABCCED")
    print(res)




