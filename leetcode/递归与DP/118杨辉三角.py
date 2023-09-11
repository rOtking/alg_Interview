class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = []
        # base case
        for i in range(1, numRows + 1):
            if i == 1:
                res.append([1])
            elif i == 2:
                res.append([1, 1])
            else:
                tmp = [0] * i
                tmp[0] = 1
                tmp[-1] = 1
                res.append(tmp)

        for i in range(2, numRows):
            # 共i+1个数，i-1个0
            for j in range(1, i):
                res[i][j] = res[i - 1][j - 1] + res[i - 1][j]

        return res


# ok
# todo 很简单，明确给出转移方程的dp，照着写就行了，就是套了个dp的壳。