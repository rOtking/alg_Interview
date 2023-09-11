import copy

class Solution:
    def __init__(self):
        self.res = []
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        # 计算首个位置的候选
        flag = self.dfs(board)
        return

    def dfs(self, board):
        if self.isOver(board):
            return True

        # 生成当前第一个空缺位置的所有可能
        candidates, x, y = self.generateProposal(board)

        flag = False
        for candidate in candidates:
            board[x][y] = candidate
            flag = self.dfs(board)
            if flag:
                break
            board[x][y] = '.'

        return flag


    def isOver(self, board):
        # 检查是否完成了
        for line in board:
            if '.' in line:
                return False

        return True

    def generateProposal(self, board):
        n = len(board[0])  # n x n
        # 初始
        candidates = [str(i) for i in range(1, n + 1)]
        # 找第一个空缺
        x, y = 0, 0
        for i in range(n):
            for j in range(n):
                if board[i][j] == '.':
                    x = i # 行
                    y = j # 列
                    break
        # 当前行所有存在的数---》去掉
        for p in board[x]:
            if p != '.' and p in candidates:
                candidates.remove(p)

        # 当前列所有存在的数---》去掉
        for line in board:
            if line[y] != '.' and line[y] in candidates:
                candidates.remove(line[y])

        # 去掉四方块的其他
        x1 = int(x / 3) * 3
        x2 = x1 + 2
        y1 = int(y / 3) * 3
        y2 = y1 + 2

        for i in range(x1, x2 + 1):
            for j in range(y1, y2 + 1):
                if board[i][j] != '.' and board[i][j] in candidates:
                    candidates.remove(board[i][j])

        return candidates, x, y


if __name__ == '__main__':
    board = [["5", "3", ".", ".", "7", ".", ".", ".", "."], ["6", ".", ".", "1", "9", "5", ".", ".", "."],
             [".", "9", "8", ".", ".", ".", ".", "6", "."], ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
             ["4", ".", ".", "8", ".", "3", ".", ".", "1"], ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
             [".", "6", ".", ".", ".", ".", "2", "8", "."], [".", ".", ".", "4", "1", "9", ".", ".", "5"],
             [".", ".", ".", ".", "8", ".", ".", "7", "9"]]

    s = Solution()
    s.solveSudoku(board)


# ok

# todo 这里的问题是，in-place的修改board，让调用函数外面的board也被修改。显然全局变量不好使了。
# todo 因为只有一个结果，那找到结果就要提前结束递归：设置个flag来控制即可！  好好消化！！hahah  不过自己确认做出来了！哈哈