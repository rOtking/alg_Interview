class Solution:
    def merge(self, A: List[int], m: int, B: List[int], n: int) -> None:
        """
        Do not return anything, modify A in-place instead.
        """
        # 从后向前  避免重叠
        p1 = m - 1
        p2 = n - 1
        p3 = len(A) - 1
        while(p1 >= 0 and p2 >= 0):
            if A[p1] > B[p2]:
                A[p3] = A[p1]
                p1 -= 1
            else:
                A[p3] = B[p2]
                p2 -= 1
            p3 -= 1

        # 剩下的不用排了
        while(p1 >= 0):
            break
        while(p2 >= 0):
            A[p3] = B[p2]
            p2 -= 1
            p3 -= 1

        return


# ok
# todo 学会变通，从后向前更简单啊！