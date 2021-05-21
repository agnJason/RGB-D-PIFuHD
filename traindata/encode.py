# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 20:52:13 2021

@author: Again Jason
One code, one world.
"""
 #i!/usr/bin/env python3
 # -*- coding:utf-8 -*-
import os 
import sys 
import codecs 
import chardet 
  
def convert(filename,out_enc="UTF-8"): 
  try: 
    content=codecs.open(filename,'rb').read()
    source_encoding='ISO-8859-9'
    print ("fileencoding:%s" % source_encoding)

    if source_encoding != None :
      content=content.decode(source_encoding).encode(out_enc) 
      codecs.open(filename,'wb').write(content)
      #content.close()
    else :
      print("can not recgonize file encoding %s" % filename)
  except IOError as err: 
    print("I/O error:{0}".format(err)) 
  
def explore(dir): 
  for root,dirs,files in os.walk(dir): 
    for file in files: 
      if os.path.splitext(file)[1]=='.obj': 
        print ("fileName:%s" % file)
        path=os.path.join(root,file)
        convert(path) 
 
def main(): 
   #explore(os.getcwd()) 
   # = input("please input dir: \n")
   explore(r'F:\zju夏令营\PIFuHD\evaldata\OBJ')
  
if __name__=="__main__": 
   main()