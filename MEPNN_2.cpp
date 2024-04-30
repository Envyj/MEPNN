#include "main.h"

// ON通道处理，亮度增
Mat On(Mat prev_frame, Mat frame) {
    //计算两个图片的像素差值
    Mat diff, result;
    /*转换为整数并两帧相减*/
    absdiff(frame, prev_frame, diff);
    //cout << diff << endl << endl;
    //用这个方法可以保留负数
    /*diff = cv::Mat(frame.size(), CV_16S);
    cv::subtract(frame, prev_frame, diff, cv::noArray(), CV_16S);*/
    //将差异中小于0的值替换为0,
    threshold(diff, result, 0, 255, THRESH_TOZERO);
    //std::cout << "result的数据如下:\n" << result << std::endl << std::endl;
    /*imshow("diff", diff);
    imshow("ON", result);
    waitKey(5);*/

    return result;;
    /*return result;*/
}
// OFF通道处理，亮度减
Mat Off(Mat prev_frame, Mat frame) {
    Mat diff, result;
    /*转换为整数并两帧相减*/
    diff = cv::Mat(frame.size(), CV_8S);
    cv::subtract(frame, prev_frame, diff);

    result = cv::Mat(diff.size(), diff.type());
    for (int r = 0; r < diff.rows; ++r) {
        for (int c = 0; c < diff.cols; ++c) {
            result.at<schar>(r, c) = (diff.at<schar>(r, c) > 0) ? 0 : std::abs(diff.at<schar>(r, c));
        }
    }
    /*imshow("OFF", result);
    waitKey(5);*/
    return result;
}

// M层，返回质心坐标
Point2f ON_OFF(Mat ON, Mat OFF) {
    Mat added_img;
    double alpha = 1.0, beta = 0.7, gamma = 0.0;

    addWeighted(ON, alpha, OFF, beta, gamma, added_img);
    //模拟弱中心环绕:通过将图像与一个从最近的相邻像素中减去10%的中心像素的内核进行卷积来模拟弱中心环绕对抗
    //定义内核
    Mat kernel = (Mat_<float>(3, 3) << -0.1, -0.1, -0.1,
        -0.1, 1.1, -0.1,
        -0.1, -0.1, -0.1);
    //将内核归一化
    kernel = kernel / sum(kernel)[0];

    //对图像进行卷积操作
    Mat filter;
    filter2D(added_img, filter, added_img.depth(), kernel);
    // 中值滤波
    //medianBlur(filter, filter, 3);
    // 二值化图像
    Mat binary;
    threshold(filter, binary, 120, 255, THRESH_BINARY);

    // 使用中值滤波器对二值化图像进行滤波，以减少噪声
    /*Mat binary_filtered;
    medianBlur(binary, binary_filtered, 3);*/

    vector<vector<Point2f>> contours;

    // 定义存储重心的变量
    cv::Point2f center;

    // 使用滤波后的图像来计算质心
    cv::Moments M = cv::moments(binary, true);


    // 计算重心
    if (M.m00 != 0) {
        center.x = static_cast<float>(M.m10 / M.m00);
        center.y = static_cast<float>(M.m01 / M.m00);
    }
    else {
        center.x = -1;
        center.y = -1;
    }

    // 绘制质心
    cv::circle(binary, center, 3, cv::Scalar(255, 255, 255), -1);
    imshow("binary", binary);
    waitKey(5);
    //return binary;
    return center;
}

void video2img_resize(string video_path)
{
    VideoCapture cap(video_path);
    ofstream out;

    //检查视频是否已打开
    if (!cap.isOpened())
    {
        cout << "Error opening video file" << endl;

    }
    //逐帧读取视频
    Mat prevFrame, frame;
    //读取第一帧
    cap >> prevFrame;


    //VideoWriter writer("output.avi", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(width, height));
    //VideoWriter videoOFF("outputOFF.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);
    resize(prevFrame, prevFrame, Size(480, 270));
    cvtColor(prevFrame, prevFrame, COLOR_BGR2GRAY);

    //对视频帧进行高斯滤波
   /*Mat blurprevImg;
   GaussianBlur(prevFrame, blurprevImg, Size(5, 5), 0, 0);*/
    vector<Mat> frames_on;
    vector<Mat> frames_off;
    vector<Mat> frames_bin;
    string fold_path = "save_img/";
    int frame_count = 0;
    //Point2f prevcentroid =(0, 0);
    vector<Point2f> feature_points;

    // 创建一个队列来保存最近的N个质心
    deque<Point2f> previous_centroids;
    const int N = 5;  // 考虑最近的5个质心进行平均
    // 初始化前一帧的质心
    Point2f prev_centroid(-1, -1);
    // 设定质心变化的最大阈值
    const double MAX_CHANGE = 100.0;
    while (true)
    {
        /*vector<Mat> channels;
        split(frame, channels);
        Mat green_channel = channels[1];
        green_channel = green_channel[:, : , - 1];绿色通道提取不太方便，放弃该方法
        waitKey(5);*/
        //读取当前帧
        cap >> frame;
        if (frame.empty())
            break;
        resize(frame, frame, Size(480, 270));
        cvtColor(frame, frame, COLOR_BGR2GRAY);

        //进行ON-OFF通路分别处理
        Mat result_ON = On(prevFrame, frame);
        //保存ON通道处理的图片结果
        frames_on.push_back(result_ON);

        // cout << result_ON.cols << ","<<result_ON.rows << endl;
        Mat result_OFF = Off(prevFrame, frame);
        frames_off.push_back(result_OFF);
        //ON_OFF(result_ON, result_OFF);
        //videoOFF.write(result_OFF);
        // 得到当前帧的质心
        Point2f centroid = ON_OFF(result_ON, result_OFF);
        /*cout << centroid << endl;*/
        // 如果保存的质心数量足够，并且当前质心与前一帧的质心的距离过大
        if (prev_centroid != Point2f(-1, -1) && norm(centroid - prev_centroid) > MAX_CHANGE) {
            // 清除之前的质心
            previous_centroids.clear();
        }

        // 更新前一帧的质心
        prev_centroid = centroid;

        // 将当前帧的质心添加到队列
        previous_centroids.push_back(centroid);

        // 如果队列中的质心数量超过了我们需要保存的数量
        if (previous_centroids.size() > N)
            // 移除最老的质心
            previous_centroids.pop_front();

        // 用平滑后的质心代替原始质心
        feature_points.push_back(centroid);

        cv::circle(frame, ON_OFF(result_ON, result_OFF), 3, cv::Scalar(255, 255, 255), -1);
        imshow("frame", frame);
        waitKey(5);
        //feature_points.push_back(ON_OFF(result_ON, result_OFF));
        //frames_bin.push_back(ON_OFF(result_ON, result_OFF));
        //将当前帧设置为前一帧
        prevFrame = frame.clone();

    }
    for (size_t i = 0; i < frames_bin.size(); ++i) {
        string file_name = fold_path + "image" + to_string(i + 1) + ".jpg";
        cv::imwrite(file_name, frames_bin[i]);
    }


    out.open("feature_points.txt");
    vector<Point2f> temp;
    //从前往后判断
    int count = 0;
    int start = -1;
    bool save_flag = false;
    for (size_t i = 0; i < feature_points.size() - 1; i++) {
        if (feature_points[i].x != -1 && feature_points[i].y != -1) {
            count++;
            if (count > 10 && !save_flag) {
                start = i - count + 1;
                save_flag = true;
            }
        }
        else {
            count = 0;
        }
    }
    //从后往前判断
    count = 0;
    for (int i = feature_points.size() - 1; i >= 0; i--) {
        if (feature_points[i].x != -1 && feature_points[i].y != -1) {
            count++;
            if (!save_flag && count > 10) {
                start = i;
                save_flag = true;
            }
        }
        else {
            count = 0;
        }
    }

    // 保存满足条件的部分
    if (save_flag) {
        int end = feature_points.size() - 1;
        while (end >= 0 && (feature_points[end].x == -1 && feature_points[end].y == -1)) {
            end--;
        }

        vector<Point2f> result(feature_points.begin() + start, feature_points.begin() + end + 1);

        vector<Point2f> filtered_result;
        filtered_result.push_back(result.back());  // 先加入最后一个点

        for (int i = result.size() - 2; i >= 0; i--) {
            double distance = norm(result[i] - filtered_result.back());
            if (distance <= 100.0) {
                filtered_result.push_back(result[i]);
            }
        }

        // 将筛选后的质心按照时间顺序重新排列
        reverse(filtered_result.begin(), filtered_result.end());

        for (size_t i = 0; i < filtered_result.size(); i++) {
            out << filtered_result[i].x << "," << filtered_result[i].y << "," << i + 1 << endl;
        }
    }
    else {
        cout << "没有找到连续质心";
    }


    out.close();
    //释放视频文件句柄
    //writer.release();
    cap.release();
}

struct DataPoint {
    double x, y;
    int frame;
};

// 在函数外部定义并初始化静态变量
static double previous_observed_x = -1;
static double previous_observed_y = -1;
std::pair<double, double> predict_next_position(double x, double y, double v, double a, double theta, double delta_t, double prev_observed_x, double prev_observed_y) {
    double delta_v = a * delta_t;
    double v_new = v + delta_v;
    double delta_p = v * delta_t + 0.5 * a * (delta_t * delta_t);
    double delta_p_x = delta_p * cos(theta);
    double delta_p_y = delta_p * sin(theta);

    // 引入状态修正因子
    const double kalman_factor = 0.5;

    // 进行修正
    if (prev_observed_x != -1 && prev_observed_y != -1) {
        double distance = norm(Point2f(x, y) - Point2f(prev_observed_x, prev_observed_y));
        double weight = exp(-distance / kalman_factor);
        x = (1 - weight) * x + weight * prev_observed_x;
        y = (1 - weight) * y + weight * prev_observed_y;
    }
    previous_observed_x = x;  // 更新上一次观测值
    previous_observed_y = y;
    return { x + delta_p_x, y + delta_p_y };
}



void calculate(std::string points_path) {
    const double fps = 30.0;
    const double delta_t = 1.0 / fps;
    std::ifstream file(points_path);

    if (!file.is_open()) {
        std::cerr << "Unable to open file." << std::endl;
        return;
    }

    std::vector<DataPoint> data;
    std::string line;

    while (getline(file, line)) {
        std::istringstream iss(line);
        DataPoint point;
        char comma;
        iss >> point.x >> comma >> point.y >> comma >> point.frame;
        data.push_back(point);
    }
    file.close();
    std::ofstream out("predicted_points.txt");  // 创建输出文件流
    for (size_t i = 1; i < data.size(); ++i) {
        double dx = data[i].x - data[i - 1].x;
        double dy = data[i].y - data[i - 1].y;

        double direction = atan2(dy, dx);  // 方向 (弧度)
        double speed = sqrt(dx * dx + dy * dy) / delta_t;  // 速度

        double acceleration = 0;
        if (i > 1) {
            double prev_speed = sqrt(pow(data[i - 1].x - data[i - 2].x, 2) + pow(data[i - 1].y - data[i - 2].y, 2)) / delta_t;
            acceleration = (speed - prev_speed) / delta_t;  // 加速度
        }

        // 预测下一个位置
        auto next_position = predict_next_position(data[i].x, data[i].y, speed, acceleration, direction, delta_t, previous_observed_x, previous_observed_y);
        // 将预测的位置点保存到文件中
        out << next_position.first << "," << next_position.second << "," << data[i].frame + 1 << std::endl;
        std::cout << "Next predicted position: (" << next_position.first << ", " << next_position.second << ")" << std::endl;

        if (i > 1) {
            std::cout << "Direction: " << direction << " rad, Speed: " << speed << " units/sec, Acceleration: " << acceleration << " units/sec^2" << std::endl;
        }
        else {
            std::cout << "Direction: " << direction << " rad, Speed: " << speed << " units/sec" << std::endl;
        }
    }
}
struct PointData {
    double x, y;
    int t;
};

std::vector<PointData> read_points_from_file(const std::string& filename) {
    std::vector<PointData> points;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return points;
    }

    std::string line;
    while (getline(file, line)) {
        PointData point;
        std::istringstream iss(line);
        char comma;
        iss >> point.x >> comma >> point.y >> comma >> point.t;
        points.push_back(point);
    }

    return points;
}
int main() {
    video2img_resize("./video/特性测试/速度测试/4.mp4");
    calculate("feature_points.txt");
    std::vector<PointData> observed_points = read_points_from_file("feature_points.txt");
    std::vector<PointData> predicted_points = read_points_from_file("predicted_points.txt");

    if (observed_points.empty() || predicted_points.empty()) {
        return 1;
    }

    cv::Mat comparison_plot = cv::Mat::zeros(800, 800, CV_8UC3);

    for (const auto& point : observed_points) {
        cv::circle(comparison_plot, cv::Point2f(point.x, point.y), 3, cv::Scalar(0, 0, 255), -1);
    }

    for (const auto& point : predicted_points) {
        cv::circle(comparison_plot, cv::Point2f(point.x, point.y), 3, cv::Scalar(0, 255, 0), -1);
    }

    cv::imshow("Observed vs Predicted", comparison_plot);
    cv::waitKey(0);

    return 0;
}