//#include "main.h"
//
//// ͼ��ҶȻ��Ͳü�
//void dispose_img(Mat& frame) {
//    cvtColor(frame, frame, COLOR_BGR2GRAY);
//    resize(frame, frame, Size(480, 270));
//    // GaussianBlur(frame, frame, Size(5, 5), 0);
//}
//
//cv::Mat processFrame(cv::Mat& frame, cv::Ptr<cv::BackgroundSubtractorMOG2>& bg_subtractor) {
//    cv::Mat fg_mask;
//    bg_subtractor->apply(frame, fg_mask);
//    cv::GaussianBlur(fg_mask, fg_mask, cv::Size(5, 5), 0);
//    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
//    cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_CLOSE, kernel);
//    return fg_mask;
//}
//
//cv::Point2f findCentroid(cv::Mat& fg_mask, cv::Mat& frame) {
//    vector<vector<cv::Point>> contours;
//    cv::findContours(fg_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//
//    // Debug: Print the number of contours and their areas
//    //cout << "Number of contours found: " << contours.size() << endl;
//
//    cv::Point2f centroid(0, 0);  // Initialize to (0, 0)
//
//    for (const auto& contour : contours) {
//        double area = cv::contourArea(contour);
//        //cout << "Contour area: " << area << endl;  // Debug: print area
//
//        if (area < 1000) {
//            continue;
//        }
//
//        cv::Moments m = cv::moments(contour);
//        //cout << "Moments: m00 = " << m.m00 << ", m10 = " << m.m10 << ", m01 = " << m.m01 << endl;  // Debug: print moments
//
//        if (m.m00 != 0) {
//            float cx = static_cast<float>(m.m10 / m.m00);
//            float cy = static_cast<float>(m.m01 / m.m00);
//            centroid = cv::Point2f(cx, cy);
//
//            // Draw centroid on frame
//            cv::circle(frame, cv::Point(cx, cy), 5, cv::Scalar(255, 255, 255), -1);
//            imshow("frame", frame);
//            waitKey(10);
//            break;  // Stop after finding the first centroid
//        }
//    }
//
//    return centroid;
//}
//void process_and_save_centroids(vector<Point2f>& feature_points) {
//    int count = 0;
//    int start = -1;
//    bool save_flag = false;
//    vector<Point2f> processed_points;
//
//    for (int i = 0; i < feature_points.size(); ++i) {
//        Point2f point = feature_points[i];
//        float x = point.x;
//        float y = point.y;
//
//        if (x != 0 || y != 0) {
//            count++;
//            if (count == 5 && !save_flag) {
//                start = i - 5;
//                cout << "Start set to: " << start << endl;  // Debug print
//                save_flag = true;
//            }
//        }
//        else {
//            count = 0;
//        }
//
//        if (save_flag) {
//            for (int j = start + 1; j < feature_points.size(); ++j) {
//                processed_points.push_back(feature_points[j]);
//            }
//            break;  // ����ѭ��
//        }
//    }
//
//    // ���ڵ��ƽ��ֵ����������
//    for (int i = 1; i < processed_points.size() - 1; ++i) {
//        Point2f point = processed_points[i];
//        Point2f prev_point = processed_points[i - 1];
//        Point2f next_point = processed_points[i + 1];
//
//        if (point.x == 0 && point.y == 0) {
//            float avg_x = (prev_point.x + next_point.x) / 2;
//            float avg_y = (prev_point.y + next_point.y) / 2;
//            processed_points[i] = Point2f(avg_x, avg_y);
//        }
//    }
//
//    // Save or print the processed points
//    ofstream outFile("processed_points.txt");
//    for (int i = 0; i < processed_points.size(); ++i) {
//        Point2f point = processed_points[i];
//        outFile << point.x << "," << point.y << "," << (i + 1) << endl;
//    }
//    outFile.close();
//}
//void saveCentroids(string video_path)
//{
//    VideoCapture cap(video_path);
//    ofstream out;
//    if (!cap.isOpened()) {
//        cout << "Error: Could not open video file." << endl;
//    }
//
//    cv::Ptr<cv::BackgroundSubtractorMOG2> bg_subtractor = cv::createBackgroundSubtractorMOG2();
//    bg_subtractor->setVarThreshold(100);
//
//    cv::Mat frame, fg_mask;
//    vector<Point2f> feature_points;
//    while (true) {
//        cap >> frame;
//
//        if (frame.empty()) {
//            break;
//        }
//        dispose_img(frame);
//        fg_mask = processFrame(frame, bg_subtractor);
//        Point2f centroid = findCentroid(fg_mask, frame);
//       
//        // Add these lines to display the frame with the drawn centroids
//        cv::imshow("Original Frame with Centroids", fg_mask);
//        cv::waitKey(30);  // Wait for 30 milliseconds
//        feature_points.push_back(centroid);     
//    }
//    process_and_save_centroids(feature_points);
//
//    out.close();
//    //�ͷ���Ƶ�ļ����
//    cap.release();
//}
//
//struct DataPoint {
//    double x, y;
//    int frame;
//};
//
////��ȡ����������������
//// ��ȡ�ļ�����ȡ����
//vector<DataPoint> read_points_from_file(const string& filename) {
//    vector<DataPoint> points;
//    ifstream file(filename);
//    if (!file.is_open()) {
//        cerr << "unable to open file: " << filename << endl;
//        return points;
//    }
//    string line;
//    while (getline(file, line)) {
//        DataPoint point;
//        istringstream iss(line);
//        char comma;
//        iss >> point.x >> comma >> point.y >> comma >> point.frame;
//        points.push_back(point);
//    }
//    return points;
//}
//pair<double, double> weighted_average(const deque<pair<double, double>>& past_observations, double decay_factor) {
//    double weight = 1.0, total_weight = 0.0;
//    double weighted_avg_x = 0.0, weighted_avg_y = 0.0;
//
//    for (const auto& obs : past_observations) {
//        weighted_avg_x += weight * obs.first;
//        weighted_avg_y += weight * obs.second;
//        total_weight += weight;
//        weight *= decay_factor;
//    }
//
//    weighted_avg_x /= total_weight;
//    weighted_avg_y /= total_weight;
//
//    return { weighted_avg_x, weighted_avg_y };
//}
//
//pair<double, double> calculatePhysics(double v, double a, double theta, double delta_t, double friction, double acceleration_factor) {
//    double non_linear_a = a - acceleration_factor * pow(v, 2);
//    double delta_v = non_linear_a * delta_t - friction * v;
//    double v_new = v + delta_v;
//    double delta_t_squared = delta_t * delta_t;
//    double delta_p = v_new * delta_t + 0.5 * non_linear_a * delta_t_squared;
//
//    return { delta_p * cos(theta), delta_p * sin(theta) };
//}
//// �����㺯��
//vector<DataPoint> calculate(const vector<DataPoint>& data, double friction, double acceleration_factor, double decay_factor) {
//    const double fps = 30.0;
//    const double delta_t = 1.0 / fps;
//    const size_t MAX_OBSERVATIONS = 5;
//    /*const double decay_factor = 0.8;*/  // ˥������
//    vector<DataPoint> predicted_data;
//
//    //cout << data.size() << endl;
//    ofstream out("predicted_points.txt");
//    if (!out.is_open()) {
//        cerr << "Unable to open output file." << endl;
//        return predicted_data;
//    }
//
//    deque<pair<double, double>> past_observations;
//
//    for (size_t i = 1; i < data.size(); ++i) {
//        double dx = data[i].x - data[i - 1].x;
//        double dy = data[i].y - data[i - 1].y;
//
//        double direction = atan2(dy, dx);  // ���� (����)
//        double speed = sqrt(dx * dx + dy * dy) / delta_t;  // �ٶ�
//
//        double acceleration = 0;
//        if (i > 1) {
//            double prev_speed = sqrt(pow(data[i - 1].x - data[i - 2].x, 2) + pow(data[i - 1].y - data[i - 2].y, 2)) / delta_t;
//            acceleration = (speed - prev_speed) / delta_t;  // ���ٶ�
//        }
//
//        // Ԥ�ⲽ��
//        auto predicted_position = calculatePhysics(speed, acceleration, direction, delta_t,friction,acceleration_factor);
//
//        // ����ǰ�۲���ӵ� past_observations
//        past_observations.push_back({ data[i].x, data[i].y });
//        if (past_observations.size() > MAX_OBSERVATIONS) {
//            past_observations.pop_front();
//        }
//
//        // ���²��裺ʹ�� past_observations �������Ȩƽ��λ��
//        auto weighted_avg = weighted_average(past_observations, decay_factor);
//
//        // ����۲�λ��
//        auto last_observation = past_observations.back();
//
//        // ������۲�λ�ÿ�ʼ�����Ԥ��λ��
//        predicted_position.first = last_observation.first + predicted_position.first;
//        predicted_position.second = last_observation.second + predicted_position.second;
//        //cout << "predicted_position-1:" << predicted_position.first << "," << predicted_position.second << "," << endl;
//        // �ü�Ȩƽ��λ������Ԥ��
//        double w1 = 0.7, w2 = 0.3;  
//
//        // �ü�Ȩƽ��λ������Ԥ��
//        predicted_position.first = w1 * predicted_position.first + w2 * weighted_avg.first;
//        predicted_position.second = w1 * predicted_position.second + w2 * weighted_avg.second;
//       /* cout << "predicted_position:" << predicted_position.first << "," << predicted_position.second << "," << endl;
//        cout << "weighted_avg:" << weighted_avg.first << "," << weighted_avg.second << "," << endl;*/
//
//        // �����������Ԥ��
//        out << predicted_position.first << "," << predicted_position.second << "," << data[i].frame + 1 << endl;
//        //cout << "predicted_position:" << predicted_position.first << "," << predicted_position.second << "," << endl;
//        predicted_data.push_back(DataPoint{ predicted_position.first, predicted_position.second, data[i].frame + 1 });
//    }
//    return predicted_data;
//}
//
//
//double calculate_error(const std::vector<DataPoint>& predicted, const std::vector<DataPoint>& actual) {
//    double error = 0.0;
//    std::unordered_map<int, DataPoint> actual_map;
//
//    // ��ʵ�����ݵ�����ϣ��
//    for (const auto& point : actual) {
//        actual_map[point.frame] = point;
//    }
//
//    // �������
//    for (const auto& pred_point : predicted) {
//        if (actual_map.find(pred_point.frame) != actual_map.end()) {
//            const auto& act_point = actual_map[pred_point.frame];
//            double dx = pred_point.x - act_point.x;
//            double dy = pred_point.y - act_point.y;
//            double current_error = std::sqrt(dx * dx + dy * dy);
//
//            // ��ӡ������Ϣ
//            /*std::cout << "Comparing frames: " << pred_point.frame << std::endl;
//            std::cout << "Predicted: (" << pred_point.x << ", " << pred_point.y << ")" << std::endl;
//            std::cout << "Actual: (" << act_point.x << ", " << act_point.y << ")" << std::endl;
//            std::cout << "Current error for this frame: " << current_error << std::endl;*/
//
//            error += current_error;
//        }
//    }
//
//    return error;
//}
//
//// ��Ѳ�������
//void grid_search() {
//    string points_path = "processed_points.txt";
//    vector<DataPoint> actual_data = read_points_from_file(points_path);
//
//    size_t N = 30;
//    vector<DataPoint> subset_actual_data(actual_data.begin(), actual_data.begin() + N);
//
//    double min_error = numeric_limits<double>::max();
//    double best_friction = 0.0;
//    double best_acceleration_factor = 0.0;
//    double best_decay_factor = 0.0;
//
//    for (double friction = 0.1; friction <= 0.9; friction += 0.1) {
//        for (double acceleration_factor = 0.05; acceleration_factor <= 0.5; acceleration_factor += 0.05) {
//            for (double decay_factor = 0.5; decay_factor <= 0.9; decay_factor += 0.1) {  
//                vector<DataPoint> predicted_data = calculate(subset_actual_data, friction, acceleration_factor, decay_factor);  
//
//                size_t subset_size = std::min(N, predicted_data.size());
//                vector<DataPoint> subset_predicted_data(predicted_data.begin(), predicted_data.begin() + subset_size);
//
//                double error = calculate_error(subset_predicted_data, subset_actual_data);
//
//                if (error < min_error) {
//                    min_error = error;
//                    best_friction = friction;
//                    best_acceleration_factor = acceleration_factor;
//                    best_decay_factor = decay_factor;  // �������˥������
//                }
//            }
//        }
//    }
//
//    cout << "Best parameters: " << endl;
//    cout << "Friction: " << best_friction << endl;
//    cout << "Acceleration Factor: " << best_acceleration_factor << endl;
//    cout << "Decay Factor: " << best_decay_factor << endl;  
//
//    // ʹ��������Ѳ������������ݽ���Ԥ��
//    vector<DataPoint> final_predicted_data = calculate(actual_data, best_friction, best_acceleration_factor, best_decay_factor);  
//}
//
//
//int main() {
//    saveCentroids("./video/���Բ���/�ԱȶȲ���/c3.mp4");
//    grid_search();
//    std::vector<DataPoint> observed_points = read_points_from_file("processed_points.txt");
//    std::vector<DataPoint> predicted_points = read_points_from_file("predicted_points.txt");
//
//    if (observed_points.empty() || predicted_points.empty()) {
//        return 1;
//    }
//
//    cv::Mat comparison_plot = cv::Mat::zeros(800, 800, CV_8UC3);
//
//    for (const auto& point : observed_points) {
//        cv::circle(comparison_plot, cv::Point2f(point.x, point.y), 3, cv::Scalar(0, 0, 255), -1);
//    }
//
//    for (const auto& point : predicted_points) {
//        cv::circle(comparison_plot, cv::Point2f(point.x, point.y), 3, cv::Scalar(0, 255, 0), -1);
//    }
//
//    cv::imshow("Observed vs Predicted", comparison_plot);
//    cv::waitKey(0);
//
//    cv::destroyAllWindows();
//    return 0;
//}
//
//
////// R�㣺��֪���ȱ仯
//////Mat R_layer(Mat prev_frame, Mat frame) {
//////    Mat diff, result;
//////    // ȷ������ͼ����16λ�з�����������
//////    prev_frame.convertTo(prev_frame, CV_16S);
//////    frame.convertTo(frame, CV_16S);
//////    // ����ԭʼ��ֵ������������
//////    subtract(frame, prev_frame, diff);
//////    //absdiff(frame, prev_frame, diff);
//////    return diff;
//////}
////////L_ON�㣬������������
//////Mat L_on(Mat frame) {
//////    Mat result = Mat::zeros(frame.size(), CV_16S);
//////    for (int r = 0; r < frame.rows; ++r) {
//////        for (int c = 0; c < frame.cols; ++c) {
//////            short val = frame.at<short>(r, c);
//////            result.at<short>(r, c) = (val > 0) ? val : 0;
//////        }
//////    }
//////    return result;
//////}
////////L_OFF�㣬�������ȼ���
//////Mat L_off(Mat frame) {
//////    Mat result = Mat::zeros(frame.size(), CV_16S);
//////    for (int r = 0; r < frame.rows; ++r) {
//////        for (int c = 0; c < frame.cols; ++c) {
//////            short val = frame.at<short>(r, c);
//////            result.at<short>(r, c) = (val > 0) ? 0 : abs(val);
//////        }
//////    }
//////    return result;
//////}
//////// L��������˷ܲ���:��ǰ֡���˷ܼ�����һ֡0.7���˷�
//////Mat lamina_excite(Mat result, Mat prev_result) {
//////    Mat excited_result;
//////    addWeighted(result, 1.0, prev_result, 0.7, 0, excited_result);
//////    return excited_result;
//////}
//////
//////Mat relu(const Mat& src) {
//////    Mat dst;
//////    max(src, 0, dst);  
//////    return dst;
//////}
//////
//////
//////// M�����������L_ON��L_OFF�˷ܲ�������������Ϊ��ǰ֡��OFF*��һ֡��ON���ϵ�ǰ֡��ON*��һ֡��OFF
//////Mat M_layer(Mat excited_result_ON, Mat excited_result_OFF, Mat excited_prev_result_ON, Mat excited_prev_result_OFF) {
//////    // ���� M ������
//////    Mat M_result = Mat::zeros(excited_result_ON.size(), excited_result_ON.type());
//////
//////    for (int r = 0; r < excited_result_ON.rows; ++r) {
//////        for (int c = 0; c < excited_result_ON.cols; ++c) {
//////            M_result.at<short>(r, c) = excited_result_OFF.at<short>(r, c) * excited_prev_result_ON.at<short>(r, c) +
//////                excited_result_ON.at<short>(r, c) * excited_prev_result_OFF.at<short>(r, c);
//////        }
//////    }
//////    M_result.convertTo(M_result, CV_8U);
//////    Mat kernel = (Mat_<short>(3, 3) << 0.25, 0.50, 0.25,
//////       0.50,1.00, 0.50,
//////       0.25, 0.50, 0.25);
//////    filter2D(M_result, M_result, M_result.depth(), kernel);
//////    M_result = relu(M_result);
//////    threshold(M_result, M_result, 150, 255, THRESH_BINARY);
//////    imshow("M", M_result);
//////    return M_result;
//////}
////
////
////
////int main() {
////    save_centroid("./video/���Բ���/�������/DJI_0764.mp4");
////    calculate("feature_points.txt");
////    vector<DataPoint> observed_points = read_points_from_file("feature_points.txt");
////        vector<DataPoint> predicted_points = read_points_from_file("predicted_points.txt");
////    
////        if (observed_points.empty() || predicted_points.empty()) {
////            return 1;
////        }
////        cv::Mat comparison_plot = cv::Mat::zeros(800, 800, CV_8UC3);
////    
////        for (const auto& point : observed_points) {
////            cv::circle(comparison_plot, cv::Point2f(point.x, point.y), 3, cv::Scalar(0, 0, 255), -1);
////        }
////    
////        for (const auto& point : predicted_points) {
////            cv::circle(comparison_plot, cv::Point2f(point.x, point.y), 3, cv::Scalar(0, 255, 0), -1);
////        }
////    
////        cv::imshow("Observed vs Predicted", comparison_plot);
////        cv::waitKey(0);
////        return 0;
////    //VideoCapture cap("./video/��Ч�Բ���/a1.mp4");
////    //ofstream out;
////    ////�����Ƶ�Ƿ��Ѵ�
////    //if (!cap.isOpened())
////    //{
////    //    cout << "Error opening video file" << endl;
////
////    //}
////    ////��֡��ȡ��Ƶ
////    //Mat prevFrame, frame;
////    ////��ȡ��һ֡
////    //cap >> prevFrame;
////    //prevFrame = dispose_img(prevFrame);
////    //while (true)
////    //{
////    //    //��ȡ��ǰ֡
////    //    cap >> frame;
////    //    if (frame.empty())
////    //           break;
////    //    frame = dispose_img(frame);
////    //    Mat diff_frame = diff(prevFrame, frame);
////    //    Point2f a = centroid(diff_frame);
////    //}
////    ////��ʼ��ǰһ֡�� ON/OFF ���
////    //Mat prev_result_ON = Mat::zeros(prevFrame.size(), CV_16S);
////    //Mat prev_result_OFF = Mat::zeros(prevFrame.size(), CV_16S);
////    //Mat prev_excited_result_ON = Mat::zeros(prevFrame.size(), CV_16S);
////    //Mat prev_excited_result_OFF = Mat::zeros(prevFrame.size(), CV_16S);
////    //while (true)
////    //{
////    //    //��ȡ��ǰ֡
////    //    cap >> frame;
////    //    if (frame.empty())
////    //        break;
////    //    frame = dispose_img(frame);
////    //    //����ON-OFFͨ·�ֱ���
////    //    Mat result_R = R_layer(prevFrame, frame);
////    //    Mat result_ON = L_on(result_R);
////    //    Mat result_OFF = L_off(result_R);
////
////    //    // ʹ��lamina_excite����
////    //    Mat excited_result_ON = lamina_excite(result_ON, prev_result_ON);
////    //    Mat excited_result_OFF = lamina_excite(result_OFF, prev_result_OFF);
////    //    //����M_layer����
////    //    Mat result_M = M_layer(excited_result_ON, excited_result_OFF, prev_excited_result_ON, prev_excited_result_OFF);
////    //    
////    //    waitKey(10);
////    //    // ����ǰһ֡�Ľ��
////    //    prev_result_ON = excited_result_ON.clone();
////    //    prev_result_OFF = excited_result_OFF.clone();
////    //    prev_excited_result_ON = excited_result_ON.clone();
////    //    prev_excited_result_OFF = excited_result_OFF.clone();
////    //    //����ǰ֡����Ϊǰһ֡
////    //    prevFrame = frame.clone();
////    //}
////    // 
////    // 
////    // 
////    // 
////    //cap.release();
////   destroyAllWindows();
////    //return 0;
////}