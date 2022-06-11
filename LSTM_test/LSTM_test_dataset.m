%rng('default');
detection = zeros(1,5);
input = zeros(1000,5);
output = zeros(1000,1);

for j = 1 : 1000
    %object_start = randi([0,22]);
    object_start = 22 * rand();
    object_vector = object_start : 1 : object_start + 5;

    for i = 1:5
        if (7 <=object_vector(1,i)) && (object_vector(1,i) <= 21)
            detection(1,i) = 1;
        else
            detection(1,i) = 0;
        end
    end

    if (7 <=object_vector(1,6)) && (object_vector(1,6)<= 21)
        out = 1;
    else
        out = 0;
    end

    input(j,:) = detection;
    output(j,1) = out;
end

csvwrite('input.csv',input)
csvwrite('output.csv',output)
