FROM python:3

ADD Linear_regression.py /

ADD supporting_functions.py /
ADD ex1data1.txt /
ADD ex1data2.txt /

RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

RUN pip install matplotlib


RUN mkdir -p ./figures
CMD [ "python", "./Linear_regression.py" ]
