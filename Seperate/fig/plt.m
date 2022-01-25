x=0:2:20;
xx=0:2:20;
% plot(x,p,'-ro','LineWidth',1.2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',6.5)
% plot(xx,pp,'-o','Color','r','LineWidth',1.2,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',6.5)
plot(x,p,'-ko','LineWidth',1,'MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',4.5)
hold on;
plot(xx,pp,'-bo','Color','b','LineWidth',1,'MarkerEdgeColor','b','MarkerFaceColor','b','MarkerSize',4.5)
plot(xx,ppp,'-ro','Color','r','LineWidth',1,'MarkerEdgeColor','r','MarkerFaceColor','r','MarkerSize',4.5)
hold off;title('Psucc as a function of SNR  in the multi-antenna scenario(N = 100, k = 4)','FontWeight','normal')
xlabel('SNR (dB)','FontSize',11)
ylabel('Psucc','FontSize',11)
grid on

legend('M=1', 'M=2','M=4','FontSize',11)



open('multiantenna1.fig');
a = get(gca,'Children');
x = get(a, 'XData');
pa = get(a, 'YData');