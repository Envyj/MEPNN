#include "main.h"

// ONͨ������������
Mat On(Mat prev_frame, Mat frame) {
    //��������ͼƬ�����ز�ֵ
    Mat diff, result;
    /*ת��Ϊ��������֡���*/
    absdiff(frame, prev_frame, diff);
    //cout << diff << endl << endl;
    //������������Ա�������
    /*diff = cv::Mat(frame.size(), CV_16S);
    cv::subtract(frame, prev_frame, diff, cv::noArray(), CV_16S);*/
    //��������С��0��ֵ�滻Ϊ0,
    threshold(diff, result, 0, 255, THRESH_TOZERO);
    //std::cout << "result����������:\n" << result << std::endl << std::endl;
    /*imshow("diff", diff);
    imshow("ON", result);
    waitKey(5);*/

    return result;;
    /*return result;*/
}
// OFFͨ���������ȼ�
Mat Off(Mat prev_frame, Mat frame) {
    Mat diff, result;
    /*ת��Ϊ��������֡���*/
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

// M�㣬������������
Point2f ON_OFF(Mat ON, Mat OFF) {
    Mat added_img;
    double alpha = 1.0, beta = 0.7, gamma = 0.0;

    addWeighted(ON, alpha, OFF, beta, gamma, added_img);
    //ģ�������Ļ���:ͨ����ͼ����һ������������������м�ȥ10%���������ص��ں˽��о����ģ�������Ļ��ƶԿ�
    //�����ں�
    Mat kernel = (Mat_<float>(3, 3) << -0.1, -0.1, -0.1,
        -0.1, 1.1, -0.1,
        -0.1, -0.1, -0.1);
    //���ں˹�һ��
    kernel = kernel / sum(kernel)[0];

    //��ͼ����о������
    Mat filter;
    filter2D(added_img, filter, added_img.depth(), kernel);
    // ��ֵ�˲�
    //medianBlur(filter, filter, 3);
    // ��ֵ��ͼ��
    Mat binary;
    threshold(filter, binary, 120, 255, THRESH_BINARY);

    // ʹ����ֵ�˲����Զ�ֵ��ͼ������˲����Լ�������
    /*Mat binary_filtered;
    medianBlur(binary, binary_filtered, 3);*/

    vector<vector<Point2f>> contours;

    // ����洢���ĵı���
    cv::Point2f center;

    // ʹ���˲����ͼ������������
    cv::Moments M = cv::moments(binary, true);


    // ��������
    if (M.m00 != 0) {
        center.x = static_cast<float>(M.m10 / M.m00);
        center.y = static_cast<float>(M.m01 / M.m00);
    }
    else {
        center.x = -1;
        center.y = -1;
    }

    // ��������
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

    //�����Ƶ�Ƿ��Ѵ�
    if (!cap.isOpened())
    {
        cout << "Error opening video file" << endl;

    }
    //��֡��ȡ��Ƶ
    Mat prevFrame, frame;
    //��ȡ��һ֡
    cap >> prevFrame;


    //VideoWriter writer("output.avi", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(width, height));
    //VideoWriter videoOFF("outputOFF.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, Size(width, height), true);
    resize(prevFrame, prevFrame, Size(480, 270));
    cvtColor(prevFrame, prevFrame, COLOR_BGR2GRAY);

    //����Ƶ֡���и�˹�˲�
   /*Mat blurprevImg;
   GaussianBlur(prevFrame, blurprevImg, Size(5, 5), 0, 0);*/
    vector<Mat> frames_on;
    vector<Mat> frames_off;
    vector<Mat> frames_bin;
    string fold_path = "save_img/";
    int frame_count = 0;
    //Point2f prevcentroid =(0, 0);
    vector<Point2f> feature_points;

    // ����һ�����������������N������
    deque<Point2f> previous_centroids;
    const int N = 5;  // ���������5�����Ľ���ƽ��
    // ��ʼ��ǰһ֡������
    Point2f prev_centroid(-1, -1);
    // �趨���ı仯�������ֵ
    const double MAX_CHANGE = 100.0;
    while (true)
    {
        /*vector<Mat> channels;
        split(frame, channels);
        Mat green_channel = channels[1];
        green_channel = green_channel[:, : , - 1];��ɫͨ����ȡ��̫���㣬�����÷���
        waitKey(5);*/
        //��ȡ��ǰ֡
        cap >> frame;
        if (frame.empty())
            break;
        resize(frame, frame, Size(480, 270));
        cvtColor(frame, frame, COLOR_BGR2GRAY);

        //����ON-OFFͨ·�ֱ���
        Mat result_ON = On(prevFrame, frame);
        //����ONͨ�������ͼƬ���
        frames_on.push_back(result_ON);

        // cout << result_ON.cols << ","<<result_ON.rows << endl;
        Mat result_OFF = Off(prevFrame, frame);
        frames_off.push_back(result_OFF);
        //ON_OFF(result_ON, result_OFF);
        //videoOFF.write(result_OFF);
        // �õ���ǰ֡������
        Point2f centroid = ON_OFF(result_ON, result_OFF);
        /*cout << centroid << endl;*/
        // �����������������㹻�����ҵ�ǰ������ǰһ֡�����ĵľ������
        if (prev_centroid != Point2f(-1, -1) && norm(centroid - prev_centroid) > MAX_CHANGE) {
            // ���֮ǰ������
            previous_centroids.clear();
        }

        // ����ǰһ֡������
        prev_centroid = centroid;

        // ����ǰ֡��������ӵ�����
        previous_centroids.push_back(centroid);

        // ��������е���������������������Ҫ���������
        if (previous_centroids.size() > N)
            // �Ƴ����ϵ�����
            previous_centroids.pop_front();

        // ��ƽ��������Ĵ���ԭʼ����
        feature_points.push_back(centroid);

        cv::circle(frame, ON_OFF(result_ON, result_OFF), 3, cv::Scalar(255, 255, 255), -1);
        imshow("frame", frame);
        waitKey(5);
        //feature_points.push_back(ON_OFF(result_ON, result_OFF));
        //frames_bin.push_back(ON_OFF(result_ON, result_OFF));
        //����ǰ֡����Ϊǰһ֡
        prevFrame = frame.clone();

    }
    for (size_t i = 0; i < frames_bin.size(); ++i) {
        string file_name = fold_path + "image" + to_string(i + 1) + ".jpg";
        cv::imwrite(file_name, frames_bin[i]);
    }


    out.open("feature_points.txt");
    vector<Point2f> temp;
    //��ǰ�����ж�
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
    //�Ӻ���ǰ�ж�
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

    // �������������Ĳ���
    if (save_flag) {
        int end = feature_points.size() - 1;
        while (end >= 0 && (feature_points[end].x == -1 && feature_points[end].y == -1)) {
            end--;
        }

        vector<Point2f> result(feature_points.begin() + start, feature_points.begin() + end + 1);

        vector<Point2f> filtered_result;
        filtered_result.push_back(result.back());  // �ȼ������һ����

        for (int i = result.size() - 2; i >= 0; i--) {
            double distance = norm(result[i] - filtered_result.back());
            if (distance <= 100.0) {
                filtered_result.push_back(result[i]);
            }
        }

        // ��ɸѡ������İ���ʱ��˳����������
        reverse(filtered_result.begin(), filtered_result.end());

        for (size_t i = 0; i < filtered_result.size(); i++) {
            out << filtered_result[i].x << "," << filtered_result[i].y << "," << i + 1 << endl;
        }
    }
    else {
        cout << "û���ҵ���������";
    }


    out.close();
    //�ͷ���Ƶ�ļ����
    //writer.release();
    cap.release();
}

struct DataPoint {
    double x, y;
    int frame;
};

// �ں����ⲿ���岢��ʼ����̬����
static double previous_observed_x = -1;
static double previous_observed_y = -1;
std::pair<double, double> predict_next_position(double x, double y, double v, double a, double theta, double delta_t, double prev_observed_x, double prev_observed_y) {
    double delta_v = a * delta_t;
    double v_new = v + delta_v;
    double delta_p = v * delta_t + 0.5 * a * (delta_t * delta_t);
    double delta_p_x = delta_p * cos(theta);
    double delta_p_y = delta_p * sin(theta);

    // ����״̬��������
    const double kalman_factor = 0.5;

    // ��������
    if (prev_observed_x != -1 && prev_observed_y != -1) {
        double distance = norm(Point2f(x, y) - Point2f(prev_observed_x, prev_observed_y));
        double weight = exp(-distance / kalman_factor);
        x = (1 - weight) * x + weight * prev_observed_x;
        y = (1 - weight) * y + weight * prev_observed_y;
    }
    previous_observed_x = x;  // ������һ�ι۲�ֵ
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
    std::ofstream out("predicted_points.txt");  // ��������ļ���
    for (size_t i = 1; i < data.size(); ++i) {
        double dx = data[i].x - data[i - 1].x;
        double dy = data[i].y - data[i - 1].y;

        double direction = atan2(dy, dx);  // ���� (����)
        double speed = sqrt(dx * dx + dy * dy) / delta_t;  // �ٶ�

        double acceleration = 0;
        if (i > 1) {
            double prev_speed = sqrt(pow(data[i - 1].x - data[i - 2].x, 2) + pow(data[i - 1].y - data[i - 2].y, 2)) / delta_t;
            acceleration = (speed - prev_speed) / delta_t;  // ���ٶ�
        }

        // Ԥ����һ��λ��
        auto next_position = predict_next_position(data[i].x, data[i].y, speed, acceleration, direction, delta_t, previous_observed_x, previous_observed_y);
        // ��Ԥ���λ�õ㱣�浽�ļ���
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
    video2img_resize("./video/���Բ���/�ٶȲ���/4.mp4");
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