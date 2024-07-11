import numpy as np
import torch
class game_agent:
    def __init__(self, input_string, status):
        self.input_string = input_string
        self.status = status
        self.verify_Matrix = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        self.verify_Matrix = np.array(self.verify_Matrix)
        
        #print(self.verify_Matrix.argmax())
    def restart(self):
        print(self.input_string)
        return('abc')
    def next_action(self):
        print('next action')
    def result(self):
        if self.input_string[1,1] == 1:
            return True
        else:
            return False
    def check_placement(self,location, weight):
        min_weight, max_weight = self.weight_ranges.get(location,(None,None))
        if min_weight is not None and max_weight is not None:
            if min_weight <= weight <= max_weight:
                return True
            else:
                return False
    def score_placement(self, location, weight):
        return 1 if self.check_placement(location, weight) else -1
    # def verify_result(self):
    #     status = False
    #     for i in range(0,8):
    #         a = self.input_string[self.verify_Matrix[i,0]]
    #         b = self.input_string[self.verify_Matrix[i,1]]
    #         c = self.input_string[self.verify_Matrix[i,2]]
    #         # print(a,b,c)
    #         if verify_3_same_value(a,b,c) == True:
    #             #print(1)
    #             #print(a,b,c)
    #             status = True
    #             break
    #     return status
    def verify_result(self):
        status = False
        if 0 not in self.input_string:
            status = True
        return status
    def verify_result_weight(self, location, weight, next_step):
        status = False
        # Define the ranges based on weight
        ranges = {
            0.5: (0, 3),
            1: (3, 6),
            1.5: (6, 9)  # Assuming 9 is the next range start or the total length
        }
        
        # Initialize mask with -inf
        mask = torch.full_like(location, -float('inf'))
        
        # Check if the intended range has space
        start, end = ranges.get(weight, (None, None))
        if start is not None and end is not None:
            mask[start:end] = location[start:end]
            if start <= next_step < end:
                status = True
            # else:
            #     # If the intended location is full, look for alternative spaces
            #     for alt_weight, (alt_start, alt_end) in ranges.items():
            #         if alt_weight != weight:  # Skip the original weight range
            #             if 0 not in location[alt_start:alt_end]:  # Check if the alternative range is full
            #                 continue  # This range is full, move to the next
            #             else:
            #                 # Found a range with space
            #                 status = True
            #                 break  # No need to check further ranges

        return status


            
