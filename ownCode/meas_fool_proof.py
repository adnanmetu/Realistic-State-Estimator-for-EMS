# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def is_read_count_line(line_data):
    if(len(line_data) <= 2):
        try:
            number = int(line_data[0:2])
            return True, number
        except ValueError:
            print("Invalid integer")
            return False , -1
    else:
        return False, -2

def read_file():
    #meas_file = "measure_deep.txt"
    meas_file = "example2measure.dat"
    with open(meas_file, "r") as file:
        lines = file.readlines()

    read_data = False
    skip_data = False
    data_record = -1
    for line in lines:
        refined_line = line.strip()
        line_data = ' '.join(refined_line.split())

        is_data_line, number = is_read_count_line(line_data)

        if is_data_line:
            print(f"number of records: {number}")
            read_data = True
            if number > 0:
                data_record = number
                skip_data = False
            else:
                skip_data = True

            continue
        if read_data and skip_data==False and data_record > 0:
            data_record -= 1
            print(line_data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    read_file()
    """
    from pyflowchart import Flowchart

    with open('metu_state_estimator_pipeline1.py') as f:
        code = f.read()
    fc = Flowchart.from_code(code)
    print(fc.flowchart())
    """
