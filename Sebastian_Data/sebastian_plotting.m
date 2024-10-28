% MATLAB script to read data from an .xlsx file and plot separate graphs

% Specify the file name
filename = '20mm 300mm per sec 30 cycles data.xlsx'; % Replace with your .xlsx file

% Read data from the .xlsx file
data = readtable(filename);

% Extract columns by their titles
test1 = data.('test1');
test2 = data.('test2');
test3 = data.('test3');

% Function to remove outliers using the IQR method
removeOutliers = @(x) x(x >= (quantile(x, 0.25) - 1.5 * iqr(x)) & x <= (quantile(x, 0.75) + 1.5 * iqr(x)));

% Remove outliers for each test
test1 = removeOutliers(test1);
test2 = removeOutliers(test2);
test3 = removeOutliers(test3);

% Plot 'test 1'
figure;
plot(test1(1:4000));
title('Test 1');
xlabel('Index');
ylabel('Value');
grid on;

% Plot 'test 2'
figure;
plot(test2(1:4000));
title('Test 2');
xlabel('Index');
ylabel('Value');
grid on;

% Plot 'test 3'
figure;
plot(test3(1:4000));
title('Test 3');
xlabel('Index');
ylabel('Value');
grid on;
