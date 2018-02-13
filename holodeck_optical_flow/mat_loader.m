clc
clear

% load pd2.mat
load pd.mat


t = 0:1/30:length(roll_c)/30;
t = t(1:end-1);

figure(1);clf
subplot(5,3,1)
plot (t,x,'r')
title('north')

subplot(5,3,2)
plot (t,y,'r')
title('east')

subplot(5,3,3)
plot (t,alt_c,'b')
hold on
plot (t,z,'r')

title('Altitude')
legend ('Commanded','Actual')

subplot(5,3,4)
plot(t,u_c,'b')
hold on
plot (t,u,'r')
title('u')

subplot(5,3,5)
plot (t,v_c,'b')
hold on
plot (t,v,'r')
title('v')

subplot(5,3,6)
hold on
plot (t,w,'r')
title('w')

subplot(5,3,7)
plot (t,-roll_c,'b')
hold on 
plot (t,roll,'r')
title('Roll')

subplot(5,3,8)
plot (t,pitch_c,'b')
hold on 
plot (t,pitch,'r')
title('Pitch')

subplot(5,3,9)
plot (t,yaw_c,'b')
hold on 
plot (t,yaw,'r')
title('Yaw')

subplot(5,3,10)
hold on 
plot (t,p,'r')
title('p')

subplot(5,3,11)
hold on 
plot (t,q,'r')
title('q')

subplot(5,3,12)
hold on 
plot (t,-yaw_rate_c,'b')
plot (t,r,'r')
title('r')

subplot(5,3,13)
hold on 
plot (t,ax,'r')
title('ax')

subplot(5,3,14)
hold on 
plot (t,ay,'r')
title('ay')

subplot(5,3,15)
hold on 
plot (t,az,'r')
title('az')

% figure(2)
% plot(y,x)
% title ('top down view of movement')
% xlabel('east')
% ylabel('north')
% 
% figure(3)
% plot3(y,x,z)
% title('flight path')
% xlabel('east')
% ylabel('north')
% zlabel('altitude')
% % view(60,-40)
% % az = 60
% % el = -40


