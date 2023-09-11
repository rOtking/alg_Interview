class Solution:
    def findLongestWord(self, s: str, dictionary) -> str:
        res = ''
        for ele in dictionary:
            if len(ele) > len(s):
                continue
            # 双指针移动
            i, j = 0, 0
            while (i < len(ele) and j < len(s)):
                if ele[i] == s[j]:
                    i += 1
                    j += 1
                else:
                    j += 1
            if i != len(ele):
                continue

            # 到这里，包含
            if len(res) == 0:
                res = ele
            else:
                if len(ele) > len(res):
                    res = ele
                elif len(ele) == len(res):
                    res = ele if ele < res else res
                else:
                    pass

        return res


# ok
# todo 当然可以先对d进行排序，再选择。

if __name__ == '__main__':
    s = "aaa"
    d = ["aaa","aa","a"]
    sol = Solution()
    res = sol.findLongestWord(s, d)
    print(res)