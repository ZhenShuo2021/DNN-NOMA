# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 09:53:59 2021

@author: Leo
"""
import NewMainBrief
from NewMainBrief import Codebook_gen
# Codebook_gen
print(__name__)
# def aa(x):
#     ...
#     return(...)

# def foo():
#     print("start...")
#     while True:
#         throw = yield 10
#         print("throw:",throw)
# g = foo()

# def z():
#     t=0
#     while t<5:
#         t = t + 1
#         print(t)
#         ans = yield t**2
#         print('ans is' , ans)
# print('__name__:', __name__)




class FooParent(object):
    def __init__(self):
        self.parent='I\'m the parent.'
        print('Parent')
        print('P1')
    def bar(self,message):
        print("%s from parent "% message)

class FooChild(FooParent):
    def __init__(self):
        # #super (FooChild,self) 首先! 會先找到 FooChild的 parent ( 就是 FooParent )，
        #然後再把FooChild的object轉換為 FooParent 的object
        super().__init__()
        print('Child')
    def bar(self,message):
        super(FooChild, self).bar(message)
        print('Child bar function')
        print(self.parent)
        return (message)


def main():
    fooChild = FooChild()
    fooChild.bar('Hello')   ## Hello from Parent

class demo:
    x=2
    def __init__(self,N, m, d):
        super().__init__(N, m ,d)
        # self.active_user = active_user
        self.N = N
        self.m = m
        # N = N
        # m = m
    def first(self):
        q = self.N+self.m
        return q
    def codebook(self):
        c = self.N*self.m
        return c
    def data(self, N, m):
        q = self.first(N,m)
        c = self.codebook(N,m)
        print(self.d)
        # q = first(self,self.N, self.m)
        # c = codebook(self,self.N,self.m)
        return q*c*demo.x


if (__name__=='__main__'):
    main()    
