def _normalize(x,min_val,max_val,new_min,new_max):
    """Map a value from one range into another using min-max scaling."""
    if min_val >= max_val:
        return new_min
    
    norm = new_min + ( (x - min_val) * (new_max - new_min) ) / (max_val - min_val)

    return norm