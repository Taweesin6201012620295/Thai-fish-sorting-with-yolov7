import numpy as np

# check range
def in_range(new_val, old_val):
    new_val, old_val = float(new_val), float(old_val)
    n = 0.05  
    if new_val >= old_val-n and new_val <= old_val+n:
        return True
    else:
        return False
    

# check same location    
def check_range(j,temp):
    x = in_range(j[1],temp[1])
    y = in_range(j[2],temp[2])
    w = in_range(j[3],temp[3])
    h = in_range(j[4],temp[4])
    #print("xywh : ",x,y,w,h)
    
    if x and y and w and h :
        #print(x,y,w,h)
        if j[5] >= temp[5]:
            #print(j[5], "max")
            return j
        elif temp[5] >= j[5]:
            #print(temp[5], "max")
            return temp
    else:
        #print("return_check_range : False")
        return False
 
# edit bboxes
def edit_boxes(arr):
    
    if len(arr) == 1:
        return arr  
    else:
        arr = np.array(arr)
        try:
            array_sort = arr[arr[:,1].argsort()]
        except:
            array_sort = arr

        get_list = []
        state = False
        check_old = None

        for i, j in enumerate(array_sort):
            
            if i==0:
                temp = np.array(array_sort[i])
            if i>0  and not(np.array_equal(j, temp)):
                #print('temp     is ', temp)
                #print('new_temp is ', j)
                check = check_range(j,temp)
                #print('check : ', check)
                #print('check_old :', check_old)
                
                if type(check) == np.ndarray:
                    if len(get_list) != 0:
                        if type(check_old) == np.ndarray and not np.array_equal(check, check_old):
                            #print("choose max conf >> pop1")
                            # check after list have conf max same now
                            new_check = check_range(check, check_old)
                            get_list.pop()
                            get_list.append(new_check.tolist())
                        else:
                            #print("choose max conf >> pop2")
                            get_list.pop()
                            get_list.append(check.tolist())
                    else: # don't have anything in list
                        #print("choose max conf >> no_pop1")
                        get_list.append(check.tolist())
                    
                else: # False
                    if len(get_list) != 0:
                        #print("this's other box1 >> new_temp")
                        get_list.append(j.tolist())
                    else:
                        #print("this's other box2 >> temp+new_temp")
                        get_list.append(temp.tolist())
                        get_list.append(j.tolist())
                        
                temp = j
                check_old = check
        return get_list


def color_class(index_class):
    match index_class:
        case 0:
            return (0,0,0) # pod is black
        case 1:
            return (255,0,0) # ku_lare is blue
        case 2:
            return (255,191,0) # see_kun is deep sky blue
        case 3:
            return (19,69,139) # too is saddle brown
        case 4:
            return (127,20,255) # khang_pan is deep pink
        case 5:
            return (0,140,255) # hang_lueang is orange
        case 6:
            return (0,0,255) # sai_dang is red
        case 7:
            return (128,0,128) # sai_dum is purple

