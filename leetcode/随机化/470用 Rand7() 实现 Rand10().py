# The rand7() API is already defined for you.
# def rand7():
# @return a random integer in the range 1 to 7

class Solution:
    def rand10(self):
        """
        :rtype: int
        """
        i = rand7()
        j = rand7()

        while (True):
            if i == 1:
                return j
            elif i == 2 and j <= 3:
                return 7 + j
            else:
                i = rand7()
                j = rand7()

# ok
# todo 等于说rand7来生成坐标，7*7的grid中每个概率都是1/49，人为定义某个位置的取之是1-10就行，只要概率一样就行！这里只用了一次1-10，其实能用
# todo 4次，命中率更好，速度也就更快！