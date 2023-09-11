class Solution:
    def hIndex(self, citations) -> int:
        if citations is None or len(citations) == 0:
            return None
        start, end = 0, len(citations) - 1
        while(start <= end):
            mid = start + int((end - start) / 2)
            if citations[mid] < len(citations) - mid:
                start = mid + 1
            else:
                end = mid - 1

        return len(citations) - start
# [0,1,3,5,6]
#  0 1 2 3 4
if __name__ == '__main__':
    s = Solution()
    res = s.hIndex(citations=[100])
    print(res)

# todo!!!! 不理解！继续！