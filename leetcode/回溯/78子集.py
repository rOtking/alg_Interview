class Solution:
    def __init__(self):
        self.res = []

    def subsets(self, nums):
        self.dfs(track=[], candidates=nums, begin=0)
        return self.res

    def dfs(self,track, candidates, begin):

        self.res.append(track[:])
        start = begin + 1
        for candidate in candidates[begin:]:
            if candidate in track:
                continue

            track.append(candidate)
            self.dfs(track, candidates, start)
            track.pop()
            start += 1
        return


if __name__ == '__main__':
    s = Solution()
    res = s.subsets([1,2,3])
    print(res)

# ok
# todo 没想到居然对了，自然终止即可！看来已经对dfs有点感觉了！哈哈 可喜可贺