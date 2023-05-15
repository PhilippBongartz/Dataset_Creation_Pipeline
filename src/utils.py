import functools
import time


runtime_dict = {}

def overall_runtime(func):
    @functools.wraps(func)
    def wrapper_record_runtime(*args, **kwargs):
        func_name = func.__name__
        if func_name not in runtime_dict:
            runtime_dict[func_name] = {'count':0,'runtime_sec':0.0}
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time -  start_time
        runtime_dict[func_name]['runtime_sec']+=runtime
        runtime_dict[func_name]['count']+=1
        return result
    return wrapper_record_runtime


    
def report_runtime_stats(func_name):
    count = runtime_dict[func_name]['count']
    summe = runtime_dict[func_name]['runtime_sec']
    mean  = summe/count
    hours = summe//3600
    minutes = (summe-hours*3600)//60
    seconds = (summe-hours*3600)-minutes*60
    overall_time = f"""{hours}h {minutes}m {seconds:.2f}s"""
    report = f"""{func_name} ran {count} times in on average {mean:.3f} seconds with overall {overall_time}"""
    print(report)
    


