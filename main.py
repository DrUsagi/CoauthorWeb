import os
import sys
import subprocess

def clear_screen():
    """清除终端屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_user_choice():
    """获取用户选择"""
    clear_screen()
    print("\n=== 学术网络分析工具 ===")
    print("\n请选择要使用的分析工具：")
    print("1. 主题网络分析")
    print("2. 研究者网络分析")
    print("0. 退出程序")
    
    while True:
        try:
            choice = input("\n请输入选项编号 [0-2]: ").strip()
            if choice in ['0', '1', '2']:
                return choice
            print("无效的选项，请重新输入！")
        except KeyboardInterrupt:
            print("\n程序已取消")
            sys.exit(0)

def get_threshold():
    """获取论文数量门槛值"""
    while True:
        try:
            threshold = input("\n请输入最小论文数量门槛值 [默认为3]: ").strip()
            if not threshold:  # 如果用户直接回车
                return 3
            threshold = int(threshold)
            if threshold < 1:
                print("门槛值必须大于0！")
                continue
            return threshold
        except ValueError:
            print("请输入有效的数字！")
        except KeyboardInterrupt:
            print("\n操作已取消")
            return 3

def check_data_file():
    """检查数据文件是否存在"""
    if not os.path.exists('llama_processed_topics.pkl'):
        print("\n错误：未找到数据文件！")
        print("请先运行 llama_topic_processor.py 生成数据。")
        return False
    return True

def main():
    if not check_data_file():
        return

    while True:
        choice = get_user_choice()
        
        if choice == '0':
            print("\n感谢使用！")
            break
            
        elif choice == '1':
            print("\n启动主题网络分析...")
            try:
                subprocess.run([sys.executable, "academic_network.py"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n运行失败：{str(e)}")
            break
            
        elif choice == '2':
            threshold = get_threshold()
            print(f"\n启动研究者网络分析 (论文数量门槛值: {threshold})...")
            try:
                env = os.environ.copy()
                env['PAPER_THRESHOLD'] = str(threshold)
                subprocess.run([sys.executable, "people_network.py"], env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n运行失败：{str(e)}")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已终止")
    except Exception as e:
        print(f"\n程序出错：{str(e)}") 