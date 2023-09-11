class Solution:
    def findContentChildren1(self, g, s) -> int:
        g = sorted(g)
        s = sorted(s)
        num = 0
        for i in g:
            to_deal = 0
            for j in s:
                if j >= i:
                    num += 1
                    to_deal = j
                    break
            if to_deal in s:
                s.remove(to_deal)
        return num

    # 双指针移动的 贪心 更快！
    def findContentChildren(self, g, s) -> int:
        g = sorted(g)
        s = sorted(s)
        num = 0
        i, j = 0, 0
        while(i < len(g) and j < len(s)):
            if g[i] <= s[j]:
                num += 1
                i += 1
                j += 1
            else:
                j += 1
        return num

# todo 虽然ok，但是双指针的贪心显然更快！！！！







if __name__ == '__main__':
    sol = Solution()
    g = [1,2,3]
    s = [1,1]
    num = sol.findContentChildren(g, s)
    print(num)