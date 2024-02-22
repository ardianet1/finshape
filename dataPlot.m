colors = {'#93B24D', '#E6E294', '#C37F95', '#79C4D1', '#38375A', ...
           '#48948B', '#135A67', '#69356E', '#9C3F2F', '#C4D9A7'};
calib = 0;
if calib
    ratio = 1;
else
    bodyLength = 129 + 85;
    pixLength = 2350;
    ratio = bodyLength/pixLength;
end
figure('Name', 'MyPlot', 'Position', [100, 100, 400, 350]);
set(gca,'linewidth',1,'fontsize',18,'fontname','Times');
opts = delimitedTextImportOptions('VariableNamesLine', 1, 'Delimiter', '\t');
data = readtable('DLTdv6_data_bodyNtailxypts.tsv', opts);
TAILFLAG = 1;
tailNum = 8;
data(1, :) = [];
dataArray = table2array(data);
dataArray(1, :) = [];
dataArray=str2double(dataArray);

frameNum = max(dataArray(:,1));
pointNum = size(dataArray,1)/frameNum/2;
posData = dataArray(:,3);
xPixPos = [];
yPixPos = [];

for i=1:pointNum
    for j=1:frameNum
        xPixPos = [xPixPos, posData(frameNum*2*(i-1) + j)];
        yPixPos = [yPixPos, posData(frameNum*2*(i-1) + frameNum + j)];
    end
end

xPixPosArray = reshape(xPixPos,frameNum,pointNum);
yPixPosArray = reshape(yPixPos,frameNum,pointNum);
hold on

straightList = [];
for i=1:4:frameNum-7
    colorIndex = mod(i,numel(colors)) + 1;
    if ismember(i,straightList)
        xOriginData = xPixPosArray(i,:);
        yOriginData = yPixPosArray(i,:);
        plot(xPixPosArray(i,:), yPixPosArray(i,:), LineWidth=3);
    else
        if TAILFLAG
            xOriginData = xPixPosArray(i,1:tailNum);
            yOriginData = yPixPosArray(i,1:tailNum);
            xOriginDataTail = xPixPosArray(i,tailNum:end);
            yOriginDataTail = yPixPosArray(i,tailNum:end);
            p = polyfit(yOriginData, xOriginData, 4);
            newY = linspace(min(yOriginData), max(yOriginData), 100); 
            newX = polyval(p, newY); 
            
            p2 = polyfit(yOriginDataTail, xOriginDataTail, 3);
            newY2 = linspace(min(yOriginDataTail), max(yOriginDataTail), 100); 
            newX2 = polyval(p2, newY2); 
            % newY = [newY, yPixPosArray(i,end)];
            % newX = [newX, xPixPosArray(i,end)];
            newY = newY * ratio - 14;
            newX = newX * ratio - 100;
            newY2 = newY2 * ratio - 14;
            newX2 = newX2 * ratio - 100;

            plot(newY, newX, LineWidth=4, Color=colors{colorIndex});
            % plot([newX(end), newX2(1)],[newY(end), newY2(1)], LineWidth=4, Color=colors{colorIndex})
            plot(newY2, newX2, LineStyle="-", LineWidth=4, Color=colors{colorIndex});
            
        else
            xOriginData = xPixPosArray(i,:);
            yOriginData = yPixPosArray(i,:);
            p = polyfit(yOriginData, xOriginData, 5);
            newY = linspace(min(yOriginData), max(yOriginData), 100); 
            newX = polyval(p, newY); 
            plot(newX, newY, LineWidth=3, Color=colors{colorIndex});
        end
    end
end
xlabel("X [mm]");
ylabel("Y [mm]");
grid on
if calib~=1
    xlim([0, 400]);
    ylim([-160, 160]);%40: Truncated, 46: Car, 52:Uta, 58:Mix, 64:Gui, 70:Oph
end
hold off
saveas(gcf, 'my_plot.png');