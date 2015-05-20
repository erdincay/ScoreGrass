__author__ = 'Kern'


class TestClass:
    def __init__(self, testStr):
        self.testStr = testStr


tc1 = TestClass("tc1")
tc2 = TestClass("tc2")
tc3 = TestClass("tc1")

testDict = {tc1: "val1", tc2: "val2", tc3: "val3"}

print(tc1, ": ", testDict[tc1])
print(tc2, ": ", testDict[tc2])
print(tc3, ": ", testDict[tc3])


testSet = {tc1, tc2, tc3}
for tc in testSet:
    print(tc)
    print(tc.testStr)
