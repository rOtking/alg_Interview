import heapq


class Solution:
    def getLeastNumbers1(self, arr: List[int], k: int) -> List[int]:
        heapq.heapify(arr)
        res = []
        while (k > 0):
            res.append(heapq.heappop(arr))
            k -= 1

        return res

    def getLeastNumbers2(self, arr: List[int], k: int) -> List[int]:
        return heapq.nsmallest(k, arr)



# ok 堆的典型应用！ez