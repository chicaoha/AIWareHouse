import numpy as np
import torch
class game_agent:
    def __init__(self, input_string, status):
        self.input_string = input_string
        self.status = status
        # self.verify_Matrix = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        # self.verify_Matrix = np.array(self.verify_Matrix)
        
    def verify_result(self):
        status = False
        if 0 not in self.input_string:
            status = True
        return status
    def verify_result_weight(self, location, weight, next_step):
        status = False
        # Define the ranges based on weight
        ranges = {
            0.5: (0, 27),
            1: (28, 69),
            1.5: (70, 97)  # Assuming 9 is the next range start or the total length
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


            
