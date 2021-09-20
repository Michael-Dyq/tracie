from itertools import count


with open(file='../../../data/uniform-prior-symbolic-format/test.txt', encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        inputs_original = []
        inputs_start = []
        inputs_duration_1 = []
        inputs_duration_2 = []

        labels_original = []
        labels_start = []
        labels_duration = []

        end_point_labels = []

        use_logic_losses = []
        use_regular_losses = []
        count = 0
        for l in lines:
            if count == 1:
                inputs_original.append(l.split("\t")[0])
                inputs_start.append(l.split("\t")[1])
                inputs_duration_1.append(l.split("\t")[2])
                inputs_duration_2.append(l.split("\t")[3])

                labels_original.append(l.split("\t")[4])
                labels_start.append("answer: positive <extra_id_2>")
                labels_duration.append("answer: <extra_id_2>")

                epl = 0
                if "ends after" in l.split("\t")[0]:
                    epl = 1
                if "positive" not in l.split("\t")[-1]:
                    epl = -100
                end_point_labels.append(epl)

                use_logic_loss = 0
                use_regular_loss = 1
                if "ends after" in l.split("\t")[0] or "ends before" in l.split("\t")[0]:
                    use_logic_loss = 1
                    use_regular_loss = 0
                use_logic_losses.append(use_logic_loss)
                use_regular_losses.append(use_regular_loss)
                break
            count += 1

        print(f'inputs_original = {inputs_original} \n \
                inputs_start = {inputs_start} \n \
                inputs_duration_1 = {inputs_duration_1} \n \
                inputs_duration_2 = {inputs_duration_2} \n \
                labels_original = {labels_original} \n \
                labels_start = {labels_start} \n \
                labels_duration = {labels_duration} \n \
                end_point_labels = {end_point_labels} \n \
                use_logic_losses = {use_logic_losses} \n \
                use_regular_losses = {use_regular_losses}')

# inputs_original = ['event: The teacher asked us to stop talking starts after we talked the whole time during and after class. \
#                     story: I was so nervous for my first day of school. "When I arrived at my first class, I was shaking." \
#                            I sat next the prettiest girl in class. We talked the whole time during and after class. "We became best friends, and she made college so much easier!"']  #why double quote
# inputs_start = ['event: The teacher asked us to stop talking starts after we talked the whole time during and after class. \
#                  story: I was so nervous for my first day of school. "When I arrived at my first class, I was shaking." \
#                         I sat next the prettiest girl in class. We talked the whole time during and after class. "We became best friends, and she made college so much easier!"']
# inputs_duration_1 = ['event:  The teacher <extra_id_1> asked us to stop talking  \
#                       story: I was so nervous for my first day of school. "When I arrived at my first class, I was shaking." I sat next the prettiest girl in class. We talked the whole time during and after class. "We became best friends, and she made college so much easier!"']
# inputs_duration_2 = ['event:  we <extra_id_1> talked the whole time during and after class . \
#                       story: I was so nervokus for my first day of school. "When I arrived at my first class, I was shaking." I sat next the prettiest girl in class. We talked the whole time during and after class. "We became best friends, and she made college so much easier!"']
# labels_original = ['answer: positive']
# labels_start = ['answer: positive <extra_id_2>']
# labels_duration = ['answer: <extra_id_2>']
# end_point_labels = [0]
# use_logic_losses = [0]  ?
# use_regular_losses = [1]


inputs_original = ['event: The teacher asked us to stop talking starts before we talked the whole time during and after class. story: I was so nervous for my first day of school. "When I arrived at my first class, I was shaking." I sat next the prettiest girl in class. We talked the whole time during and after class. "We became best friends, and she made college so much easier!"']
inputs_start = ['event: The teacher asked us to stop talking starts before we talked the whole time during and after class. story: I was so nervous for my first day of school. "When I arrived at my first class, I was shaking." I sat next the prettiest girl in class. We talked the whole time during and after class. "We became best friends, and she made college so much easier!"']
inputs_duration_1 = ['event:  The teacher <extra_id_1> asked us to stop talking  story: I was so nervous for my first day of school. "When I arrived at my first class, I was shaking." I sat next the prettiest girl in class. We talked the whole time during and after class. "We became best friends, and she made college so much easier!"']
inputs_duration_2 = ['event:  we <extra_id_1> talked the whole time during and after class .  story: I was so nervous for my first day of school. "When I arrived at my first class, I was shaking." I sat next the prettiest girl in class. We talked the whole time during and after class. "We became best friends, and she made college so much easier!"']
labels_original = ['answer: negative']
labels_start = ['answer: positive <extra_id_2>']  #?
labels_duration = ['answer: <extra_id_2>']
end_point_labels = [-100]
use_logic_losses = [0]
use_regular_losses = [1]

