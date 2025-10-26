import SwiftUI

struct LoginView: View {
    @StateObject private var vm = LoginViewModel()
    var onSuccess: () -> Void

    var body: some View {
        NavigationStack {
            VStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Welcome back")
                        .font(.largeTitle.bold())
                    Text("Sign in to sync your plans and groceries.")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                VStack(spacing: 12) {
                    TextField("Email", text: $vm.email)
                        .textContentType(.username)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .textFieldStyle(.roundedBorder)

                    SecureField("Password", text: $vm.password)
                        .textContentType(.password)
                        .textFieldStyle(.roundedBorder)
                }

                if let err = vm.errorMessage {
                    Text(err)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                Button {
                    vm.login { onSuccess() }
                } label: {
                    HStack {
                        if vm.isLoading { ProgressView().tint(.white) }
                        Text(vm.isLoading ? "Signing in…" : "Sign In")
                            .bold()
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                }
                .buttonStyle(.borderedProminent)
                .disabled(vm.isLoading)

                Spacer()
            }
            .padding(24)
            .navigationTitle("Login")
        }
    }
}

#Preview {
    LoginView(onSuccess: {})
}

