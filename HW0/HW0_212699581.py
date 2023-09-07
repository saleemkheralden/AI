
# Q1
# since sorted works in O(nlogn)
# and the while loop is O(n)
# the total time complexity is O(nlogn)
def twoSum(nums: list[int], target: int) -> list[int]:
    N = len(nums)
    numsSorted = sorted(nums)
    s = 0
    e = N - 1
    while s < e:
        se = numsSorted[s] + numsSorted[e]
        if se == target:
            return [s, e]
        elif se > target:
            e -= 1
        elif se < target:
            s += 1
    return None

# Q2
def maxProfit(prices: list[int]) -> int:
    e = len(prices) - 1
    s = 0
    ret = 0
    for e in range(len(prices) - 1, -1, -1):
        for s in range(e):
            ret = max(ret, prices[e] - prices[s])
    return ret

# Q3
class Node:
    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node
    
    def __str__(self):
        return f"{self.value}->{self.next}"

# Q3.1
def read_file(file_path: str) -> Node:
    with open(file_path) as file:
        a = file.readline().split(';')
        head = Node(int(a[0]))
        p = head
        for e in a[1:]:
            p.next = Node(int(e))
            p = p.next
        return head

# Q3.2
def get_length(head: Node) -> int:
    r = 0
    while head is not None:
        r += 1
        head = head.next
    return r

# Q3.3
# time complexity: O(nlogn)
# space complexity: O(n)
def sort_in_place(head: Node) -> Node:
    a = []
    p = head
    while p is not None:
        a.append(p.value)
        p = p.next

    a = sorted(a)
    p = head

    for e in a:
        p.value = e
        p = p.next
    
    return head



# print(maxProfit([7, 1, 5, 3, 6, 4]))
# print(twoSum([2, 8, 11, 15], 28))
# print(sort_in_place(read_file("test.txt")))

