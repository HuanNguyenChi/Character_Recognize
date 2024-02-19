Giới thiệu chung
Machine learning là một nhánh của trí tuệ nhân tạo (AI) và khoa học máy tính, tập trung vào việc sử dụng dữ liệu và thuật toán để bắt chước hành động của con người, dần dần cải thiện độ chính xác của nó.

Machine learning còn là một thành phần quan trọng của lĩnh vực khoa học dữ liệu đang phát triển. Thông qua việc sử dụng các phương pháp thống kê, các thuật toán được đào tạo để đưa ra các phân loại hoặc dự đoán và khám phá những thông tin chi tiết từ chính các dự án khai thác dữ liệu.

Thông qua các thông tin chi tiết có được để thúc đẩy việc đưa ra quyết định đối với các ứng dụng và doanh nghiệp, tác động mạnh đến các chỉ số tăng trưởng. Khi dữ liệu lớn tiếp tục nhu cầu mở rộng và phát triển đòi hỏi nhu cầu tuyển dụng các nhà khoa học dữ liệu sẽ tăng lên. Họ sẽ được yêu cầu giúp xác định các câu hỏi kinh doanh có liên quan nhất và dữ liệu để trả lời chúng.

Bài toán của machine learning thường được chia làm hai loại là dự đoán (prediction) và phân loại (classification). Các bài toán dự đoán thường là giá nhà, giá xe, v.v, còn các bài toán phân loại thường là nhận diện chữ viết tay, đồ vật, v.v.

Trong project này, chúng ta sẽ tìm hiểu về bài toán phân loại(classification), cụ thể là dự đoán chữ cái viết tay trong bảng Alphabets. Trong phần này ta sẽ sử dụng phương pháp Supervised learning (Học có giám sát), kết hợp với mạng nơ-ron tích chập để giải quyết với độ chính xác khá cao.
 
Mục Lục
I.	Giới thiệu về mạng nơ-ron tích chập (CNN)	4
1.	Convolution Layer	4
2, Hàm phi tuyến ReLU	5
3.	Lớp Gộp(Pooling Layer)	6
4.	Fully connected layer	7
II.	Xử lí bài toán	8
1.	Tiền sử lí dữ liệu	8
1.1	Giới thiệu về tập dữ liệu	8
1.2	Xáo trộn tập dữ liệu	9
1.3	Phân chia tập dữ liệu	10
1.4	Mã Hóa One-Hot	11
2.	Xây dựng model	13
2.1	Các khái niệm liên quan	13
2.2	Huấn luyện mô hình	16
2.3	Đánh giá mô hình	20
2.4	Dự đoán	20
2.5	Lưu mô hình	22
III.	Sử dụng mô hình để nhận diện ảnh và chữ viết tay	22
3.1	Xây dựng module Predict	22
3.2	Xây dựng Winform App	23
3.3	Hướng dẫn sử dụng	26
IV.	Tài liệu tham khảo	28
 
I Giới thiệu về mạng nơ-ron tích chập (CNN)
Convolutional Neural Network (CNNs – Mạng nơ-ron tích chập) là một trong những mô hình Deep Learning tiên tiến. Nó giúp cho chúng ta xây dựng được những hệ thống thông minh với độ chính xác cao như hiện nay. Cũng chính vì tính ứng dụng cao nó đã trở thành công cụ giúp nhận dạng vật thể – object trong ảnh
Vd: như hệ thống xử lý ảnh lớn như facebook, google hay amazon đã đưa vào sản phẩm của mình những chức năng thông minh như nhận diện khuôn mặt người dùng, phát triển xe hơi tự lái hay drone giao hàng tự động.
*Cấu trúc mạng 1 mạng CNN:


Mạng CNN là một tập hợp các lớp Convolution chồng lên nhau và sử dụng các hàm nonlinear activation như ReLU và tanh để kích hoạt các trọng số trong các node. Ngoài ra có một số lớp khác như pooling/subsampling layer dùng để chắt lọc lại các thông tin hữu ích hơn (loại bỏ các thông tin nhiễu). Mỗi một lớp sau khi thông qua các hàm kích hoạt sẽ tạo ra các thông tin trừu tượng hơn cho các lớp tiếp theo.

1.	Convolutional layer (Lớp tích chập)
Convolution layer là lớp quan trọng nhất và cũng là lớp đầu tiên của của mô hình CNN. Lớp này có chức năng chính là phát hiện các đặc trưng có tính không gian hiệu quả. Trong tầng này có 4 đối tượng chính là: ma trận đầu vào, bộ filters, và receptive field, feature map. Conv layer nhận đầu vào là một ma trận 3 chiều và một bộ filters cần phải học. Bộ filters này sẽ trượt qua từng vị trí trên bức ảnh để
 
tính tích chập (convolution) giữa bộ filter và phần tương ứng trên bức ảnh. Phần tưng ứng này trên bức ảnh gọi là receptive field, tức là vùng mà một neuron có thể nhìn thấy để đưa ra quyết định, và mà trận cho ra bởi quá trình này được gọi là feature map

*Stride và padding
Stride: số pixel thay đổi trên ma trận đầu vào. Khi stride là 1 thì ta di chuyển các kernel 1 pixel. Khi stride là 2 thì ta di chuyển các kernel đi 2 pixel và tiếp tục như vậy. Hình dưới là lớp tích chập hoạt động với stride là 2
Padding: mặc định là 0, đươẹ sử dụng khi ma trận đầu vào không phù hợp với kernel.
2.	Hàm phi tuyến ReLU
ReLU viết tắt của Rectified Linear Unit, là 1 hàm phi tuyến. Với đầu ra là: ƒ
(x) = max (0, x), thường được sử dụng sau lớp Conv.
Đây chính là một hàm kích hoạt trong Neural Network. Chúng ta có thể biết đến hàm kích hoạt này với một tên gọi khác là Activation Function. Nhiệm vụ chính của hàm kích hoạt là mô phỏng lại các Neuron có tỷ lệ truyền xung qua Axon. Trong đó, hàm kích hoạt sẽ bao gồm các hàm cơ bản như: Sigmoid, Tanh, Relu, Leaky Relu, Maxout.
 
Hiện nay, hàm Relu đang được sử dụng khá phổ biến và thông dụng. Đặc biệt, Relu sở hữu những ưu điểm nổi bật như: hỗ trợ tính toán nhanh nên rất được ưa chuộng sử dụng trong việc huấn luyện các mạng Neuron.


3.	Lớp Gộp (Pooling layer)

Lớp pooling sẽ giảm bớt số lượng tham số khi hình ảnh quá lớn. Không gian pooling còn được gọi là lấy mẫu con hoặc lấy mẫu xuống làm giảm kích thước của mỗi map nhưng vẫn giữ lại thông tin quan trọng. Các pooling có thể có nhiều loại khác nhau

•	Max Pooling
•	Average Pooling
•	Sum Pooling


 
Ý tương đằng sau tầng pooling là vị trí tuyết đối của những đặc trưng trong không gian ảnh không còn cần cần thiết, thay vào đó vị trí tương đối giữ các đặc trưng đã đủ để phân loại đối tượng. Hơn giảm tầng pooling có khả năng giảm chiều cực kì nhiều, làm hạn chế overfit, và giảm thời gian huấn luyện tốt.

Đối với các loại ảnh thông thường sẽ có kích thước là 2×2, tuy nhiên nếu đầu vào hình ảnh của bạn lớn thì có thể sử dụng Pooling Size 4×4 để đảm bảo chất lượng cho ảnh.

4.	Fully Connected Layer

Sau khi ảnh được truyền qua nhiều convolutional layer và pooling layer thì model đã học được tương đối các đặc điểm của ảnh thì tensor của output của layer cuối cùng sẽ được là phẳng thành vector và đưa vào một lớp được kết nối như một mạng nơ-ron. Với FC layer được kết hợp với các tính năng lại với nhau để tạo ra một mô hình. Cuối cùng sử dụng softmax hoặc sigmoid để phân loại đầu ra.

Ví dụ, trong bài toán phân loại số viết tay MNIST có 10 lớp tương ứng 10 số từ 0-9, tầng fully connected layer sẽ chuyển ma trận đặc trưng của tầng trước thành vector có 10 chiều thể hiện xác suất của 10 lớp tương ứng.


5.	Tóm tắt

•	Đầu vào của lớp tích chập là hình ảnh
•	Chọn đối số, áp dụng các bộ lọc với các bước nhảy, padding nếu cần. Thực hiện tích chập cho hình ảnh và áp dụng hàm kích hoạt ReLU cho ma trận hình ảnh.
 
•	Thực hiện Pooling để giảm kích thước cho hình ảnh.
•	Thêm nhiều lớp tích chập sao cho phù hợp
•	Xây dựng đầu ra và dữ liệu đầu vào thành 1 lớp được kết nối đầy đủ (Full Connected)
•	Sử dụng hàm kích hoạt để tìm đối số phù hợp và phân loại hình ảnh.

II, Xử lí bài toán
Import các thư viện cần thiết:


1.	Tiền xử lí dữ liệu (Data pre-processing)
1.1.	Giới thiệu về tập dữ liệu
_ Data set là tập bao gồm 26 chữ cái Tiếng Anh viết tay được lấy trên [Kaggle].
https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in- csv-format
_ Data bao gồm hơn 370.000 bức ảnh kích thước 28*28, ký tự đầu tiên đại diện cho kí tự mà nó thuộc về
•	A  ‘0’
•	B  ‘1’
•	….
•	Z  ‘25’
_ Do đó độ dài của list đại diện, tức mỗi dòng dữ liệu là: 28 * 28 +1 = 785. Ta tiến hành đọc tập dữ liệu đã tải xuống.
 


 

Dữ liệu bao gồm 372450 bức ảnh được số hóa


_ Ta thử in ra bức ảnh đầu tiên để kiểm tra:




_ Kí tự đầu tiên là ‘0’ tương ứng với ký tự A






1.2.	Xáo trộn tập dữ liệu
_ Chia tập dữ liệu ban đầu thành 2 tập y (label) và X(dữ liệu bức ảnh)

 

_ Mục đích của việc xáo trộn dữ liệu là để dữ liệu training hoàn toàn là ngẫu nhiên, không theo một thứ tự nào, từ đó cải thiện quá trình đào tạo
_ Các giá trị pixel có thể nằm trong khoảng từ 0 đến 255. Mỗi số đại diện cho một mã màu, để tránh việc tính toán với trị số cao chúng ta có thể chuyển các giá trị về phạm vi từ 0  1 mà không làm thay đổi tính chất của bức ảnh. Vì các giá trị nằm trong khoảng [0, 255]  chia các giá trị cho 255.0


_ Tiến hành xáo trộn dữ liệu


1.3	Phân chia dữ liệu
_ Để chuẩn bị cho bài toán Machine Learning, ta tiến hành chia tập dữ liệu thành
training set, validation set và testing set
•	Training Set là là một tập dữ liệu dùng để huấn luyện cho mô hình của thuật toán Machine Learning.
•	Testing set là tập dùng để dánh giá, kiểm nghiệm mô hình vừa huấn luyện có hiệu quả hay không
 
•	Validation set là tập các giá trị input đi kèm với giá trị output và được dùng để kiểm thử độc chính xác của mô hình máy học trong quá trình huấn luyện


•	Vậy sự khác nhau giữa công dụng Testing Set và Validation Set là gì ? Testing được dùng để kiểm thử sau quá trình huấn luyện, còn validation set được sử dụng để kiểm thử trong quá trình huấn luyện. Chính vì vậy, thuật ngữ overfitting (hiện tượng mô hình dự đoán quá khớp với tập training set, dẫn đến dự đoán không hiệu quả đối với tập testing set.) cần phải nắm rõ trong quá trình sử dụng. Thông thường, người ta ngầm cho rằng Validation set mà có độ chính xác cao thì tập Testing set cũng có độ chính xác cao


_ Ta tiến hành chia dữ liệu:


1.4	Mã hóa One-hot (One-hot Encoding)
 
_ Trong cách mã hóa này, một “từ điển” cần được xây dựng chứa tất cả các giá trị khả dĩ của từng dữ liệu hạng mục. Sau đó mỗi giá trị hạng mục sẽ được mã hóa bằng một vector nhị phân với toàn bộ các phần tử bằng 0 trừ một phần tử bằng 1 tương ứng với vị trí của giá trị hạng mục đó trong từ điển.
_ Trong ví dụ về đặc trưng màu sắc, đặc trưng này có 3 giá trị rời rạc và do vậy chúng ta sẽ biến đổi đặc trưng này thành 3 đặc trưng nhị phân đồng thời đặc trưng màu sắc bị xóa bỏ. Một cáchtổng quát hóa, mã hóa one-hot sẽ cần n đặc trưng mới để lưu trữ giá trị cho một đặc trưng nhóm có n giá trị rời rạc




_ Trước khi thực hiện mã hóa ta thay đổi kích cỡ các tập dữ liệu để chuẩn bị đưa vào mô hình đào tạo


 
_ Ta thực hiện mã hóa bằng hàm to_categorical() trong thư viện keras.untils:


2.	Xây dựng Model
2.1	Các khái niệm liên quan
2.1.1	Keras
Keras là một API cấp cao được thiết kế cho Python để triển khai mạng nơ- ron dễ dàng hơn. Nó được phát triển bởi Google và có thể chạy trên các thư viện và khung công tác như TensorFlow, Theano, PlaidML, MXNet, CNTK. Chúng đều là những thư viện rất mạnh nhưng cũng khó hiểu để tạo mạng nơ-ron. Mặt khác, Keras rất thân thiện với người mới bắt đầu vì cấu trúc tối thiểu của nó cung cấp cách tạo ra các mô hình học sâu một cách dễ dàng và gọn ghẽ dựa trên TensorFlow hoặc Theano.

Những lý do nên sử dụng Keras để bắt đầu:
•	Keras ưu tiên trải nghiệm của người lập trình
•	Keras đã được sử dụng rộng rãi trong doanh nghiệp và cộng đồng nghiên cứu
•	Keras giúp dễ dàng biến các thiết kế thành sản phẩm
•	Keras hỗ trợ huấn luyện trên nhiều GPU phân tán
•	Keras hỗ trợ đa backend engines và không giới hạn bạn vào một hệ sinh thái

2.1.2	Sequencial()
API mô hình tuần tự (sequential) là đơn giản nhất và là API được khuyên dùng, đặc biệt là khi mới bắt đầu. Nó được gọi là “sequential” vì nó liên quan đến việc xác định một lớp Sequential và thêm từng lớp vào mô hình theo cách tuyến tính, từ đầu vào đến đầu ra.
 

2.1.3	Conv2d ()
Conv2D là convolution dùng để lấy feature từ ảnh với các tham số :
•	filters : số filter của convolution
•	kernel_size : kích thước window search trên ảnh
•	strides : số bước nhảy trên ảnh
•	activation : chọn activation như linear, softmax, relu, tanh, sigmoid
•	padding : có thể là "valid" hoặc "same". Với same thì có nghĩa là padding =1
•	input_shape: kích thước ma trận đầu vào

2.1.4	Maxpooling2d ()
Pooling Layer thực hiện các hoạt động tổng hợp tối đa cho dữ liệu, giảm bớt tham số nhưng vẫn giữ lại những feature của ảnh
•	Pool_size: kích thước kernel, thường là (2,2) hoặc (4,4)
•	Strides: bước nhảy, mặc định sẽ bằng với kích thước pool_size

2.1.5	Flatten()
Lớp làm phẳng được sử dụng để làm phẳng đầu vào bằng cách không ảnh hưởng đến kích thước


 
2.1.6	Dense()
_ Thể hiện một fully connected layer, tức toàn bộ các unit của layer trước đó được nối với toàn bộ các unit của layer hiện tại, với các tham số cơ bản
•	Units: số chiều output
•	Activation: chọn các hàm kích hoạt (ReLU, softmax, sigmoid….)
_ Hàm activation Softmax Function dịch ra Tiếng Việt là hàm trung bình mũ. Nó tính toán xác suất xảy ra của một sự kiện. Nói một cách khái quát, hàm softmax sẽ tính khả năng xuất hiện của một class trong tổng số tất cả các class có thể xuất hiện. Sau đó, xác suất này sẽ được sử dụng để xác định class mục tiêu cho các input.






2.1.7	Compile ()
_ Được dùng để định cấu hình quá trình học tập của mô hình
_ Các tham số quan trọng:
•	Loss function: Trong học máy, hàm Loss được sử dụng để tìm lỗi hoặc sai lệch trong quá trình học, một số hàm mất mát như (mean_squared_error, categorical_crossentropy….)
•	Optimizer (Tối ưu) : Tối ưu hóa là một quá trình quan trọng nhằm tối ưu các trọng số đầu vào bằng cách so sánh dự đoán và hàm mất mát. Keras
 
cung cấp khá nhiều trình tối ưu hóa dưới dạng mô-đun (SGD, Adam, Nadam,….)
•	Metrics: Chỉ số được sử dụng để đánh giá hiệu suất của mô hình . Nó tương tự như hàm mất mát, nhưng không được sử dụng trong quá trình đào tạo, một số metrics thường dung ( accurency, clone_metric)

2.1.8	fit()
Các mô hình được đào tạo bởi mảng NumPy bằng cách sử dụng fit (). Mục đích chính của hàm này được sử dụng để đánh giá mô hình của bạn khi đào tạo. Điều này cũng có thể được sử dụng để vẽ đồ thị hiệu suất mô hình
•	Bao gồm data train, test đưa vào training.
•	Batch_size thể hiện số lượng mẫu mà Mini-batch sử dụng cho mỗi lần cập nhật trọng số .
•	Epoch là số lần duyệt qua hết số lượng mẫu trong tập huấn luyện.

2.2	Tiến hành huấn luyện mô hình
_ Lần lượt thêm các lớp và các hàm activation:


_ Theo dõi Output và lượng param cần học sau từng lớp bằng keras.sumary():
 
 

_ Sơ đồ hoạt động:
•	Công thức tính số tham số(param) của một mạng phức hợp:

Channel_in * kernel_width * kernel_height * chanel_out + chanel_out

•	Công thức tính size của Output_shape:


 
 Size = n_out * n_out * filter



 
28*28*1
 

n_in = 28, k = 3, p =1, s = 1 -> n_out =28
param = 1* 3 * 3 * 32 + 32 = 320
 
28*28*32
 






 
28*28*32
 

n_in = 28, k = 3, p =1, s = 1 -> n_out =28
param = 32 * 3 * 3 * 32 + 32 = 9248
 





 


n_in = 28, k = 2, p =0, s = 2 -> n_out =14
 
14*14*32
 






 
14*14*64
 


n_in = 14, k = 3, p =1, s = 1 -> n_out =14
param = 32 * 3 * 3 * 64 + 64 = 18496
 





 

n_in = 14, k = 3, p =1, s = 1 -> n_out =14
param = 64 * 3 * 3 * 64 + 64 = 36928
 
14*14*64
 




 

7*7*64
 
n_in = 14, k = 2, p =0, s = 2 -> n_out =7
 
 

Param = 128 * 26 + 26 = 3354


 Total params = 1740154

_ Biên dịch model
 
_ Tiến hành train model:


_ Quá trình đào tạo:
 

2.3	Đánh giá mô hình

_ Đánh giá là một quá trình trong quá trình phát triển mô hình để kiểm tra xem liệu mô hình có phù hợp nhất với vấn đề đã cho và dữ liệu tương ứng hay không. Keras cung cấp hàm evaluate() và nó có ba đối số chính,
•	Test data
•	Test data label
•	verbose - true hay false
_ Tiến hành đánh giá mô hình:

_ Độ chính xác là: 99, 06 %
2.4	Dự đoán
 
_ Tiến hành dự đoán một số bộ test:

_ Kết quả:
 
2.5	Lưu mô hình

_Trong	trường	hợp	muốn	lấy	model	ra	đẻ	sử	dụng	sử	dụng	hàm keras.load_model()

 
III.	Sử Dụng Model Để Tiến Hành Nhận Diện File Ảnh Và Chữ Viết Tay
3.1	Xây dựng Module “Predict”
_ Import các thư viện cần thiết:

_ Load model và khai báo 1 dictionary chứa các class của đầu ra:


_ Trong module, xây dựng hàm sol(path) để xử lí ảnh đầu vào sao cho phù hợp với module để tiến hành dự đoán (với path là đường dẫn của ảnh đầu vào):

_ Hàm sol() trả về kí tự có tỷ lệ chính xác cao nhất
 
 
3.2	Xây dựng Winform App
_ Import các thư viện cần thiết:

_ Tạo các hàm chức năng cơ bản:
•	browse_file(): Load file từ máy tính
•	detect(): Thực hiện quá trình dự đoán
•	clear(): Xóa đi input cũ để thực hiện
•	paint(); Vẽ kí tự cần detect()

 
_ Tiến hành cài đặt giao diện:


 
 

 
 

3.3	Hướng dẫn sử dụng
_ Run file app.py:

_ Giao diện của app

 
_ Tiến hành vẽ thử 1 ký tự tại box Detect:

_ Kiểm tra kết quả:
 

_ Click ‘Clear’ để tiếp tục dự đoán
_ Tương tự, ta có thể browse những bức ảnh bằng ‘Update’:
 
_ Kết quả:


_ Tại tab Terminal, chúng ta có thể xem rõ độ chính xác của từng kí tự:


IV.	Tài liệu tham khảo
https://github.com/KudoKhang/Hand-Written https://keras.io/api/models/sequential/ https://keras.io/api/models/model_training_apis/
https://v1study.com/python-tham-khao-tach-tap-du-lieu-cua-ban-voi- train_test_split-cua-scikit-learning.html
https://viblo.asia/p/nhan-dang-chu-cai-viet-tay-su-dung-deep-learning-
GrLZDwNJKk0
https://v1study.com/python-mo-hinh-tuan-tu-sequential-model.html
