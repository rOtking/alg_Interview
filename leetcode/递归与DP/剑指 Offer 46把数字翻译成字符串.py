class Solution:
    def translateNum(self, num: int) -> int:
        # todo 1(0-9)，2(0-5),3-9与0单独
        chs = str(num)
        return self.process(chs, 0)

    def process(self, chs, i):
        if i == len(chs) - 1:
            return 1

        res = 0
        # todo 至少2个数 i+1一定存在
        if chs[i] == '1':
            res = self.process(chs, i + 1)
            if i + 2 <= len(chs) - 1:
                res += self.process(chs, i + 2)
            else:
                res += 1

        elif chs[i] == '2':
            res = self.process(chs, i + 1)
            if int(chs[i + 1]) <= 5:
                if i + 2 <= len(chs) - 1:
                    res += self.process(chs, i + 2)
                else:
                    res += 1
            else:
                pass


        else:
            res = self.process(chs, i + 1)

        return res


# ok
# 分类讨论，没什么问题
# todo 重点是边界与i+2的问题。