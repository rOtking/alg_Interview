class Solution:
    def __init__(self):
        self.res = []

    def letterCombinations(self, digits: str):
        '''
        就是个DFS，下面就是每一个的可选择列表
        :param digits:
        :return:
        '''
        if digits is None or len(digits) == 0:
            return self.res
        digit2phone = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }

        arr = []
        for digit in digits:
            arr.append(digit2phone[digit])

        self.dfs(track='', candidates=arr, begin=0)
        return self.res

    def dfs(self, track, candidates, begin):
        if len(track) == len(candidates):
            self.res.append(track[:])
            return

        # 记住，这就是当前的选择
        for i in candidates[begin]:
            track = track + i  # str
            self.dfs(track, candidates, begin + 1)
            track = track[:-1]
        return





if __name__ == '__main__':
    s = Solution()
    res = s.letterCombinations('23')
    print(res)


# ok
# todo dfs就是梳理清楚每一步的选择列表与当前的track