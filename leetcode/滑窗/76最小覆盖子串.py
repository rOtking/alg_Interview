class Solution:

    def minWindow(self, s: str, t: str) -> str:
        if len(s) == 1 and s == t:
            return s
        # 构建需求，因为与个数有关系。即还需多少改元素
        need = {}
        totalNeed = 0   # 总共需要多少
        for i in t:
            if i in need:
                need[i] += 1
            else:
                need[i] = 1
            totalNeed += 1

        l, r = 0, 0
        res = s
        while(r < len(s)):
            if s[r] in need and need[s[r]] > 0:
                need[s[r]] -= 1
                totalNeed -= 1

            while(totalNeed == 0):
                if r - l + 1 < len(res):
                    res = s[l : r + 1]
                if s[l] in need:
                    need[s[l]] += 1
                    totalNeed += 1
                l += 1
            r += 1

        return res


if __name__ == '__main__':
    s = Solution()
    res = s.minWindow(s="ADOBECODEBANC", t="ABC")
    print(res)

# todo 滑窗的核心思想：1.不满足要求时右边界拼命的右移，知道满足条件；2.满足条件后，左边界右移，拼命压缩到最小的满足长度；3.左边移动到不满足后，右边再寻找....
