from operator import length_hint
from src.services.evaluation import Evaluation
from src.services.analysis import Analysis
from io import TextIOWrapper
from typing import Dict, List
import numpy as np
import itertools
import argparse
import pickle
import copy
import os

def removeComments(file_:TextIOWrapper):
    f = file_.read().replace(' ', '').replace('\n', '').replace('\t', '')            
    open_comment = f.find('#', 0)            
    while open_comment != -1:        
        end_comment = f.find('#', open_comment + 1)        
        f = f.replace(f[open_comment:end_comment + 1], '')
        open_comment = f.find('#', open_comment)                
    return f

def convertNumber(dict_list:Dict[str, any]):    
    for key, values in dict_list.items():            
        new_values = []
        for x in values:
            try:                
                new_values.append(int(x)) if float(x) == int(float(x)) else new_values.append(float(x))
            except ValueError:
                new_values.append(x)
        dict_list[key] = new_values
    return dict_list

def getKey(file_, i):
    idx_colon = file_.find(':', i)           
    var_name = file_[i:idx_colon]       
    i = idx_colon + 1
    return var_name, i

def getValues(file_, i):
    idx_semicolon = file_.find(';', i)
    idx_comma = file_.find(',', i)                    
    values = []
    while idx_comma < idx_semicolon and idx_comma != -1:
        idx_quote = file_.find("'", i)                
        i = idx_quote + 1
        idx_quote = file_.find("'", i)
        values.append(file_[i:idx_quote])                
        i = idx_quote + 1        
        idx_comma = file_.find(',', i)                                      
    if file_[i:idx_semicolon] != '':                 
        i+=1
        idx_quote = file_.find("'", i)                                
        if idx_quote < idx_semicolon: values.append(file_[i:idx_quote]) 
        else: values.append(file_[i:idx_semicolon])
    i = idx_semicolon + 1
    return values, i

def toDictionary(path:str, args:list):            
    file_list = []
    for files in args:            
        variables = {}                
        f = open(f'{path}{files}', mode='r')
        clean_f = removeComments(f)        
        f.close()
        key, i = getKey(clean_f, 0)            
        while i < len(clean_f) and i > 0:
            values, i = getValues(clean_f, i)                        
            variables.update({key : values})                                              
            key, i = getKey(clean_f, i)            
        variables = convertNumber(variables)        
        file_list.append(variables)            
    return file_list

def dynamicReduction(i, indexes:List[int], reset:List[int]):    
    if indexes[i] > 0:
        indexes[i] -= 1
    else :
        if i - 1 >= 0:
            dynamicReduction(i - 1, indexes, reset)
            indexes[i] = reset[i]    
        else:
            indexes[i] -= 1

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default=f'.{os.sep}config{os.sep}', help='path where config files are')
    parser.add_argument('-f', '--files', type=str, nargs='+', required=True, help='files to load parameters args;')
    
    args = vars(parser.parse_args())
    dict_list = toDictionary(args['path'], args['files'])    
    
    for file_ in dict_list:
        temp_keys = ''
        constant_keys = []
        for key, value in file_.items():            
            if len(value) == 1:                
                temp_keys+= f'--{key} {value[0]} '       
                constant_keys.append(key)                

        for key in constant_keys:
            file_.pop(key)        
        names, values = list(file_.keys()), list(file_.values())                      
        
        del constant_keys
        del file_
        del dict_list
        del args
        del parser

        indexes = [len(x) - 1 for x in values]
        reset = copy.deepcopy(indexes)
        run = f'cmd /c py "{os.curdir}{os.sep}Neural Network{os.sep}main.py"'    
        total = len(values) - 1       
        while indexes[0] >= 0:                        
            args = ''
            for i in range(len(values)):
                args+= f'--{names[i]} {values[i][indexes[i]]} '                                        
            os.system(f'{run} {temp_keys} {args}')                    
            dynamicReduction(total, indexes, reset)            

    #Analysis.best_thresolds()
    #Evaluation.best_results()
