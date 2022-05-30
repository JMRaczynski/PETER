def are_explanations_same(t1, t2):
    return t1.rindex(" ") == t2.rindex(" ") and t1[:t1.rindex(" ")] == t2[:t1.rindex(" ")]


path_1 = "../models/Yelp/generated_rating_input_yelp_no_teacher_force.txt"
path_2 = "../models/Yelp/generated_rating_input_yelp.txt"
lines = [3925, 4693, 7149, 9078, 11035, 12562, 14346, 34326, 34981, 37277, 39921, 44606, 44658, 46009, 48642, 69129, 82865, 91174, 97331, 114029, 117077, 119503, 121053, 124225, 124777, 125803, 127114, 128113, 133939, 136942, 149245, 153041, 155530, 157689, 158262, 163897, 178349, 181025, 181745, 184153, 185062, 188695, 189814, 199450, 201367, 204070, 206341, 209345, 214221, 218153, 220513, 224401, 232221, 245097, 249433, 250187, 253623, 262353, 263265, 272563, 277686, 284121, 296174, 296669, 296686, 305259, 305929, 314570, 335459, 339765, 343515, 346353, 359643, 366238, 369143, 375611, 375915, 379179, 380106, 393315, 398398, 402610, 407854, 411783, 415862, 418957, 422517, 428837, 431257, 444805, 447517, 454359, 456226, 457893, 459962, 466393, 466945, 476899, 501217, 516165]
for line_no, ind in enumerate(lines):
    lines[line_no] = lines[line_no] + (3 - lines[line_no] % 4)

with open(path_1, "r") as f:
    no_tf = f.readlines()

with open(path_2, "r") as f:
    tf = f.readlines()

counter = 0
for line_no in lines:
    if not are_explanations_same(no_tf[line_no - 1], tf[line_no - 1]):
        counter += 1
        print(line_no - 1, "\n", no_tf[line_no - 1], tf[line_no - 1], sep="")

print(counter)
