class Solution:
    def relativeSortArray(self, arr1, arr2):
        ex = []   # 额外数组
        num = [0] * len(arr2)
        for i in range(len(arr1)):
            if arr1[i] in arr2:
                index = arr2.index(arr1[i])
                num[index] += 1
            else:
                ex.append(arr1[i])

        ex = sorted(ex)
        res = []
        for i in range(len(arr2)):
            tmp = [arr2[i]] * num[i]
            res.extend(tmp)

        res.extend(ex)

        return res



# acc 没问题！